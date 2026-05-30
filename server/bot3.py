#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bright Smile Dental memory-first front-desk voice agent.

A caller reaches the dental front desk by browser WebRTC or Twilio phone call.
Appointment and insurance tools are backed by the in-memory mock backend in
``tools.py``.

Pipeline: Nemotron Speech Streaming STT -> memory-first front desk processor
-> Nemotron-3-Super-120B LLM fallback -> Gradium TTS, with direct function
tools registered on the LLM context.

Run the bot using::

    uv run bot.py
"""

import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import (
    RunnerArguments,
    SmallWebRTCRunnerArguments,
    WebSocketRunnerArguments,
)
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.gradium.tts import GradiumTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from metrics import LatencyLogger, StageLatencyLogger
from nemotron_llm import VLLMOpenAILLMService
from nvidia_stt import NVidiaWebSocketSTTService
from tools import (
    TOOL_IMPLS,
    build_system_instruction,
    pipecat_tools_schema,
    register_pipecat_functions,
)

load_dotenv(override=True)

WEBRTC_AUDIO_IN_SAMPLE_RATE = 16000
WEBRTC_AUDIO_OUT_SAMPLE_RATE = 24000
TWILIO_AUDIO_IN_SAMPLE_RATE = 16000
TWILIO_AUDIO_OUT_SAMPLE_RATE = 8000
STT_SAMPLE_RATE = 16000

DEFAULT_TWILIO_LOOKUP_TIMEOUT_SECS = 0.5
DEFAULT_NEMOTRON_LLM_TEMPERATURE = 0.2
DEFAULT_NEMOTRON_LLM_MAX_TOKENS = 240

DEFAULT_TWILIO_VAD_CONFIDENCE = 0.55
DEFAULT_TWILIO_VAD_START_SECS = 0.12
DEFAULT_TWILIO_VAD_STOP_SECS = 0.25
DEFAULT_TWILIO_VAD_MIN_VOLUME = 0.35

DEFAULT_WEBRTC_VAD_CONFIDENCE = 0.55
DEFAULT_WEBRTC_VAD_START_SECS = 0.12
DEFAULT_WEBRTC_VAD_STOP_SECS = 0.55
DEFAULT_WEBRTC_VAD_MIN_VOLUME = 0.35

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return float(raw_value)
    except ValueError:
        logger.warning("{}={} is not a valid float; using {}", name, raw_value, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("{}={} is not a valid integer; using {}", name, raw_value, default)
        return default


def _next_weekday(today: date, weekday: int) -> date:
    days = (weekday - today.weekday()) % 7
    if days == 0:
        days = 7
    return today + timedelta(days=days)


def _normalize_text(text: str) -> str:
    normalized = (
        text.replace("\u2019", "'")
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u202f", " ")
        .replace("\xa0", " ")
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _extract_relative_or_absolute_date(text: str, today: date) -> str | None:
    lowered = _normalize_text(text).lower()
    if re.search(r"\btomorrow\b", lowered):
        return (today + timedelta(days=1)).isoformat()

    for weekday_name, weekday in WEEKDAYS.items():
        if f"this {weekday_name}" in lowered or f"next {weekday_name}" in lowered:
            return _next_weekday(today, weekday).isoformat()
        if re.search(rf"\b{weekday_name}\b", lowered):
            return _next_weekday(today, weekday).isoformat()

    month_match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(\d{1,2}|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
        r"(?:,?\s*(\d{4}))?",
        lowered,
    )
    if not month_match:
        return None

    month_names = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    ordinal_days = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    raw_day = month_match.group(2)
    day = int(raw_day) if raw_day.isdigit() else ordinal_days[raw_day]
    year = int(month_match.group(3) or today.year)
    return date(year, month_names[month_match.group(1)], day).isoformat()


def _extract_time(text: str) -> str | None:
    lowered = _normalize_text(text).lower().replace(".", "")
    if re.search(r"\b(1|one)\s*(pm|p m)\b", lowered):
        return "1:00 PM"
    if re.search(r"\b(2|two)\s*(pm|p m)\b", lowered):
        return "2:00 PM"
    if re.search(r"\b(2:30|two thirty)\s*(pm|p m)?\b", lowered):
        return "2:30 PM"
    if re.search(r"\b(4|four)\s*(pm|p m)\b", lowered):
        return "4:00 PM"
    return None


def _clean_name(candidate: str) -> str | None:
    words = [word.strip(" .,!?:;").lower() for word in candidate.split()]
    words = [word for word in words if word]
    if not words:
        return None

    non_names = {
        "a",
        "an",
        "the",
        "new",
        "patient",
        "caller",
        "calling",
        "appointment",
        "cleaning",
        "check",
        "checkup",
        "cavity",
        "visit",
        "and",
        "at",
        "for",
        "because",
        "with",
        "hi",
        "hello",
        "hey",
        "aria",
        "arya",
    }
    if words[0] in non_names:
        return None
    trimmed = []
    for word in words:
        if word in non_names:
            break
        trimmed.append(word)
    if not trimmed or len(trimmed) > 3:
        return None
    return " ".join(word.capitalize() for word in trimmed)


def _extract_name(text: str) -> str | None:
    normalized = _normalize_text(text)
    patterns = [
        r"\bmy name is ([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
        r"\bname is ([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
        r"\bthis is ([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
        r"\bi'?m ([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
        r"\bi am ([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            cleaned = _clean_name(match.group(1))
            if cleaned:
                return cleaned
    leading_name = re.match(r"\s*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})[.,!?]\s+", normalized)
    if leading_name:
        cleaned = _clean_name(leading_name.group(1))
        if cleaned:
            return cleaned
    bare_name = re.fullmatch(r"\s*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\.?\s*", normalized)
    if bare_name:
        return _clean_name(bare_name.group(1))
    return None


def _extract_reason(text: str) -> str | None:
    lowered = _normalize_text(text).lower()
    if "cleaning" in lowered:
        return "routine cleaning"
    if "cavity" in lowered:
        return "cavity follow-up"
    if "sensitivity" in lowered:
        return "tooth sensitivity"
    if "new patient" in lowered or "first visit" in lowered:
        return "new patient visit"
    if "check-up" in lowered or "checkup" in lowered or "check up" in lowered:
        return "check-up"
    if "chipped filling" in lowered:
        return "chipped filling"
    return None


def _extract_confirmation_id(text: str) -> str | None:
    match = re.search(r"\b(BSD\s*\d{3,6})\b", _normalize_text(text), re.IGNORECASE)
    if not match:
        return None
    return re.sub(r"\s+", "", match.group(1)).upper()


def _mentions_vague_afternoon(text: str) -> bool:
    return bool(re.search(r"\b(afternoon|after lunch|later today)\b", _normalize_text(text).lower()))


def _looks_like_correction(text: str) -> bool:
    lowered = _normalize_text(text).lower()
    return any(
        phrase in lowered
        for phrase in ("actually", "sorry", "correction", "correct that", "i mean", "instead", "rather")
    )


def _format_fast_date(iso_date: str) -> str:
    parsed = date.fromisoformat(iso_date)
    return parsed.strftime("%A, %B %d").replace(" 0", " ")


def _format_fast_time(time_value: str) -> str:
    return {
        "1:00 PM": "one PM",
        "2:00 PM": "two PM",
        "2:30 PM": "two thirty PM",
        "4:00 PM": "four PM",
    }.get(time_value, time_value)


def _format_slot_list(slots: list[str]) -> str:
    spoken = [_format_fast_time(slot) for slot in slots]
    if len(spoken) == 1:
        return spoken[0]
    return ", ".join(spoken[:-1]) + f", or {spoken[-1]}"


@dataclass
class CallMemory:
    """Short-lived per-call memory. It is never persisted across calls."""

    intent: str = ""
    name: str = ""
    appointment_date: str = ""
    appointment_time: str = ""
    reason: str = ""
    reschedule_confirmation_id: str = ""
    last_question: str = ""
    confirmation_id: str = ""
    vague_time_requested: bool = False
    wants_time_options: bool = False
    offered_slots: list[str] = field(default_factory=list)

    def has_booking_context(self) -> bool:
        return any(
            (
                self.intent == "booking",
                self.appointment_date,
                self.appointment_time,
                self.reason,
                self.vague_time_requested,
                self.wants_time_options,
                self.offered_slots,
            )
        )

    def booking_ready(self) -> bool:
        return all((self.name, self.appointment_date, self.appointment_time, self.reason))


class MemoryFirstFrontDeskProcessor(FrameProcessor):
    """Handles common front-desk turns from state before falling back to the LLM."""

    def __init__(self, *, today: date | None = None):
        super().__init__()
        self._today = today or date.today()
        self._memory = CallMemory()
        self._processed_user_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM or not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        response = self._response_for_context(frame.context)
        if response is None:
            await self.push_frame(frame, direction)
            return

        await self.push_frame(LLMFullResponseStartFrame(), direction)
        await self.push_frame(LLMTextFrame(response), direction)
        await self.push_frame(LLMFullResponseEndFrame(), direction)

    def _response_for_context(self, context: Any) -> str | None:
        new_texts = self._new_caller_texts(context)
        if not new_texts:
            return None

        for text in new_texts:
            self._remember(text)

        latest = _normalize_text(new_texts[-1])
        lowered = latest.lower()

        if self._is_greeting_only(lowered) and not self._memory.intent:
            return None

        policy_response = self._policy_response(lowered)
        if policy_response:
            return policy_response

        if self._looks_like_insurance(lowered):
            return self._insurance_response(lowered)

        if self._looks_like_reschedule(lowered) or self._memory.intent == "reschedule":
            self._memory.intent = "reschedule"
            return self._reschedule_response()

        if self._looks_like_booking(lowered) or self._memory.has_booking_context():
            self._memory.intent = "booking"
            return self._booking_response(lowered)

        return None

    def _new_caller_texts(self, context: Any) -> list[str]:
        user_texts = []
        for message in context.get_messages():
            if message.get("role") != "user" or not isinstance(message.get("content"), str):
                continue
            content = message["content"]
            if "A caller just reached the dental front desk" in content:
                continue
            user_texts.append(content)

        new_texts = user_texts[self._processed_user_count :]
        self._processed_user_count = len(user_texts)
        return new_texts

    def _remember(self, text: str) -> None:
        normalized = _normalize_text(text)
        lowered = normalized.lower()
        is_correction = _looks_like_correction(normalized)

        if self._looks_like_booking(lowered):
            self._memory.intent = "booking"
        if self._looks_like_reschedule(lowered):
            self._memory.intent = "reschedule"

        name = _extract_name(normalized)
        if name:
            self._memory.name = name

        appointment_date = self._date_from_text(normalized)
        if appointment_date and (not self._memory.appointment_date or is_correction or self._memory.intent):
            self._memory.appointment_date = appointment_date

        appointment_time = _extract_time(normalized)
        if appointment_time:
            self._memory.appointment_time = appointment_time
            self._memory.vague_time_requested = False

        reason = _extract_reason(normalized)
        if reason:
            self._memory.reason = reason

        confirmation_id = _extract_confirmation_id(normalized)
        if confirmation_id:
            self._memory.reschedule_confirmation_id = confirmation_id

        if _mentions_vague_afternoon(normalized) and not appointment_time:
            self._memory.vague_time_requested = True
        if any(phrase in lowered for phrase in ("what times", "which times", "options", "open slots")):
            self._memory.wants_time_options = True

    def _date_from_text(self, text: str) -> str | None:
        lowered = _normalize_text(text).lower()
        if "after that" in lowered and self._memory.appointment_date:
            try:
                return (date.fromisoformat(self._memory.appointment_date) + timedelta(days=7)).isoformat()
            except ValueError:
                return None
        return _extract_relative_or_absolute_date(text, self._today)

    def _booking_response(self, lowered: str) -> str:
        if self._is_confirmation(lowered) and self._memory.offered_slots and not self._memory.appointment_time:
            self._memory.appointment_time = self._preferred_slot(self._memory.offered_slots)
            self._memory.vague_time_requested = False

        if not self._memory.name:
            self._memory.last_question = "name"
            return "May I have your name?"
        if not self._memory.reason:
            self._memory.last_question = "reason"
            return "What is the reason for the visit?"
        if not self._memory.appointment_date:
            self._memory.last_question = "date"
            return "What date would you like?"
        if not self._memory.appointment_time:
            if self._memory.vague_time_requested and not self._memory.wants_time_options:
                slots = self._available_slots()
                self._memory.appointment_time = self._preferred_slot(slots) if slots else "2:00 PM"
                self._memory.vague_time_requested = False
                return self._book_from_memory()
            if self._memory.vague_time_requested or self._memory.offered_slots:
                self._memory.last_question = "time"
                return self._offer_slots()
            self._memory.last_question = "time"
            return "What time would you like?"

        return self._book_from_memory()

    def _available_slots(self) -> list[str]:
        result = TOOL_IMPLS["check_availability"]({"date": self._memory.appointment_date})
        raw_slots = result.get("open_slots") or []
        return [slot for slot in raw_slots if isinstance(slot, str)]

    def _offer_slots(self) -> str:
        self._memory.offered_slots = self._available_slots()
        if not self._memory.offered_slots:
            return "I do not see open slots that day. Would another date work?"
        return (
            f"I have {_format_slot_list(self._memory.offered_slots)} on "
            f"{_format_fast_date(self._memory.appointment_date)}. Which works?"
        )

    def _book_from_memory(self) -> str:
        booking = {
            "name": self._memory.name,
            "date": self._memory.appointment_date,
            "time": self._memory.appointment_time,
            "reason": self._memory.reason,
        }
        result = TOOL_IMPLS["book_appointment"](booking)
        confirmation_id = str(result.get("confirmation_id", ""))
        self._memory.confirmation_id = confirmation_id
        self._memory.intent = ""
        self._memory.last_question = ""
        self._memory.offered_slots = []
        self._memory.wants_time_options = False
        return (
            f"Booked for {_format_fast_date(result['date'])} at {_format_fast_time(result['time'])}. "
            f"Confirmation {confirmation_id}."
        )

    def _reschedule_response(self) -> str:
        if not self._memory.reschedule_confirmation_id:
            self._memory.last_question = "confirmation_id"
            return "What is your confirmation ID?"
        if not self._memory.appointment_date:
            self._memory.last_question = "date"
            return "What date should I move it to?"
        if not self._memory.appointment_time:
            self._memory.last_question = "time"
            return "What time should I move it to?"

        result = TOOL_IMPLS["reschedule_appointment"](
            {
                "confirmation_id": self._memory.reschedule_confirmation_id,
                "date": self._memory.appointment_date,
                "time": self._memory.appointment_time,
            }
        )
        self._memory.intent = ""
        if result.get("status") == "rescheduled":
            return (
                f"You're rescheduled for {_format_fast_date(result['date'])} "
                f"at {_format_fast_time(result['time'])}."
            )
        return "I could not find that confirmation ID. The office can help look it up."

    def _insurance_response(self, lowered: str) -> str:
        provider = self._extract_insurance_provider(lowered)
        result = TOOL_IMPLS["check_insurance"]({"provider": provider})
        if result.get("accepted"):
            return f"Yes, Bright Smile Dental accepts {provider}."
        return f"The office will confirm {provider} coverage for you."

    def _policy_response(self, lowered: str) -> str | None:
        if any(word in lowered for word in ("severe pain", "facial swelling", "trauma", "bleeding heavily")):
            return "Please seek emergency care first. I can also help schedule an urgent dental visit."
        if any(phrase in lowered for phrase in ("ibuprofen", "root canal", "diagnose", "what dose")):
            return "I cannot give dental or medication advice, but I can help book a dentist visit."
        if "cancel" in lowered:
            return "I cannot cancel appointments here. The office can help, or I can help reschedule."
        if "hours" in lowered or "closes" in lowered or "open today" in lowered:
            return "The office will confirm current hours for you."
        if "phone number" in lowered or "pull up my chart" in lowered or "know who i am" in lowered:
            return "I cannot identify you or pull up records from caller ID alone."
        if "goodbye" in lowered or re.fullmatch(r"(bye|thanks bye|thank you bye)[.!]?", lowered):
            return "Thanks for calling Bright Smile Dental. Goodbye."
        return None

    def _looks_like_booking(self, lowered: str) -> bool:
        intent_words = (
            "schedule",
            "book",
            "appointment",
            "visit",
            "cleaning",
            "follow-up",
            "follow up",
            "check-up",
            "checkup",
            "chipped filling",
            "first visit",
        )
        return any(word in lowered for word in intent_words) and "cancel" not in lowered

    def _looks_like_reschedule(self, lowered: str) -> bool:
        return any(word in lowered for word in ("reschedule", "move", "change my appointment"))

    def _looks_like_insurance(self, lowered: str) -> bool:
        providers = ("delta dental", "metlife", "aetna", "cigna", "guardian")
        return "insurance" in lowered or "coverage" in lowered or "covered" in lowered or any(
            provider in lowered for provider in providers
        )

    def _extract_insurance_provider(self, lowered: str) -> str:
        for provider in ("Delta Dental", "MetLife", "Aetna", "Cigna PPO", "Cigna", "Guardian"):
            if provider.lower() in lowered:
                return provider
        return "that plan"

    def _is_confirmation(self, lowered: str) -> bool:
        return any(word in lowered for word in ("yes", "works", "sounds good", "please", "book it"))

    def _is_greeting_only(self, lowered: str) -> bool:
        return bool(re.fullmatch(r"(hi|hello|hey|hi aria|hi arya)[.!]?", lowered))

    def _preferred_slot(self, slots: list[str]) -> str:
        return "2:00 PM" if "2:00 PM" in slots else slots[0]


def twilio_transport_overrides() -> dict[str, int]:
    return {
        "audio_in_sample_rate": TWILIO_AUDIO_IN_SAMPLE_RATE,
        "audio_out_sample_rate": TWILIO_AUDIO_OUT_SAMPLE_RATE,
    }


def build_twilio_vad_params() -> VADParams:
    return VADParams(
        confidence=_env_float("VOICE_VAD_CONFIDENCE", DEFAULT_TWILIO_VAD_CONFIDENCE),
        start_secs=_env_float("VOICE_VAD_START_SECS", DEFAULT_TWILIO_VAD_START_SECS),
        stop_secs=_env_float("VOICE_VAD_STOP_SECS", DEFAULT_TWILIO_VAD_STOP_SECS),
        min_volume=_env_float("VOICE_VAD_MIN_VOLUME", DEFAULT_TWILIO_VAD_MIN_VOLUME),
    )


def build_webrtc_vad_params() -> VADParams:
    return VADParams(
        confidence=_env_float("VOICE_WEBRTC_VAD_CONFIDENCE", DEFAULT_WEBRTC_VAD_CONFIDENCE),
        start_secs=_env_float("VOICE_WEBRTC_VAD_START_SECS", DEFAULT_WEBRTC_VAD_START_SECS),
        stop_secs=_env_float("VOICE_WEBRTC_VAD_STOP_SECS", DEFAULT_WEBRTC_VAD_STOP_SECS),
        min_volume=_env_float("VOICE_WEBRTC_VAD_MIN_VOLUME", DEFAULT_WEBRTC_VAD_MIN_VOLUME),
    )


def build_vad_analyzer(*, twilio: bool, audio_in_sample_rate: int) -> SileroVADAnalyzer:
    return SileroVADAnalyzer(
        sample_rate=audio_in_sample_rate,
        params=build_twilio_vad_params() if twilio else build_webrtc_vad_params(),
    )


def build_user_aggregator_params(vad_analyzer: SileroVADAnalyzer) -> LLMUserAggregatorParams:
    return LLMUserAggregatorParams(vad_analyzer=vad_analyzer)


def build_stt_service() -> NVidiaWebSocketSTTService:
    return NVidiaWebSocketSTTService(
        url=os.getenv("NVIDIA_ASR_URL", "ws://192.168.7.228:8081"),
        sample_rate=STT_SAMPLE_RATE,
        strip_interim_prefix=True,
    )


def build_twilio_serializer(call_data: dict[str, str]) -> TwilioFrameSerializer:
    return TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        params=TwilioFrameSerializer.InputParams(
            twilio_sample_rate=TWILIO_AUDIO_OUT_SAMPLE_RATE,
            sample_rate=TWILIO_AUDIO_IN_SAMPLE_RATE,
        ),
    )


async def get_call_info(call_sid: str, timeout_secs: float | None = None) -> dict:
    """Fetch call information from Twilio REST API using aiohttp.

    Args:
        call_sid: The Twilio call SID

    Returns:
        Dictionary containing call information including from_number, to_number, status, etc.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        logger.warning("Missing Twilio credentials, cannot fetch call info")
        return {}

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json"

    timeout_secs = (
        timeout_secs
        if timeout_secs is not None
        else _env_float("TWILIO_CALLER_INFO_TIMEOUT_SECS", DEFAULT_TWILIO_LOOKUP_TIMEOUT_SECS)
    )

    try:
        # Use HTTP Basic Auth with aiohttp
        auth = aiohttp.BasicAuth(account_sid, auth_token)
        timeout = aiohttp.ClientTimeout(total=timeout_secs)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, auth=auth) as response:
                if response.status != 200:
                    logger.warning("Twilio caller-info lookup failed with status {}", response.status)
                    return {}

                data = await response.json()

                call_info = {
                    "from_number": data.get("from"),
                    "to_number": data.get("to"),
                }

                return call_info

    except TimeoutError:
        logger.warning("Twilio caller-info lookup timed out after {:.2f}s", timeout_secs)
        return {}
    except Exception as e:
        logger.warning("Twilio caller-info lookup failed: {}", e.__class__.__name__)
        return {}


async def run_bot(
    transport: BaseTransport,
    from_number: str | None = None,
    audio_in_sample_rate: int = WEBRTC_AUDIO_IN_SAMPLE_RATE,
    audio_out_sample_rate: int = WEBRTC_AUDIO_OUT_SAMPLE_RATE,
    twilio: bool = False,
):
    """Main bot logic.

    Args:
        transport: The transport to use.
        from_number: Caller's phone number (Twilio path only) for logging context.
        audio_in_sample_rate: Input audio sample rate in Hz. Defaults to 16000 (WebRTC).
        audio_out_sample_rate: Output audio sample rate in Hz. Defaults to 24000 (WebRTC).
    """
    logger.info("Starting bot3")

    tools = pipecat_tools_schema()

    system_instruction = build_system_instruction(from_number=from_number)

    # Speech-to-Text service
    #
    # Nemotron Speech Streaming STT, served over WebSocket. The server expects
    # 16-bit PCM, 16 kHz, mono — matching the WebRTC input path. The URL can be
    # overridden via NVIDIA_ASR_URL.
    stt = build_stt_service()
    stt_sample_rate = getattr(stt, "_init_sample_rate", stt.sample_rate)
    logger.info(
        "Active audio sample rates input_hz={} output_hz={} stt_hz={}",
        audio_in_sample_rate,
        audio_out_sample_rate,
        stt_sample_rate,
    )

    # LLM service — Nemotron-3-Super-120B served by vLLM (OpenAI-compatible chat
    # completions at /v1). vLLM exposes the Chat Completions API, not the Responses
    # API, so we use OpenAILLMService (not OpenAIResponsesLLMService). The live
    # endpoint serves the model as "nemotron-3-super" (per its /v1/models).
    #
    # Reasoning ("thinking") toggle — Nemotron is controlled per-request via
    # chat_template_kwargs.enable_thinking, forwarded through the OpenAI client's
    # extra_body (the request-body convention confirmed against this endpoint in
    # ../aiewf-eval traces). Default OFF for low-latency voice. To ENABLE, set
    # NEMOTRON_ENABLE_THINKING=true; to DISABLE, leave unset/false.
    #
    # CAUTION for voice: reasoning is only kept out of the spoken `content` if the
    # vLLM server runs a reasoning parser (e.g. --reasoning-parser nemotron_v3, which
    # routes it to a separate `reasoning_content` field). This live endpoint did NOT
    # surface reasoning_content in testing, so if thinking is enabled and the server
    # lacks a parser, chain-of-thought would appear inline in `content` and get
    # spoken. Keep thinking OFF for voice unless the parser is confirmed active.
    # VLLMOpenAILLMService is a thin OpenAILLMService subclass that reports TTFB to
    # the first NON-THINKING token (so the metric reflects time-to-first-spoken-word
    # when reasoning is enabled, not time-to-first-reasoning-token). No-op when
    # thinking is off. See server/nemotron_llm.py.
    enable_thinking = os.getenv("NEMOTRON_ENABLE_THINKING", "false").lower() == "true"
    llm = VLLMOpenAILLMService(
        api_key=os.getenv("NEMOTRON_LLM_API_KEY", "EMPTY"),  # vLLM ignores unless --api-key set
        base_url=os.getenv("NEMOTRON_LLM_URL", "http://192.168.7.228:8000/v1"),
        settings=VLLMOpenAILLMService.Settings(
            model=os.getenv("NEMOTRON_LLM_MODEL", "nvidia/nemotron-3-super"),
            system_instruction=system_instruction,
            temperature=_env_float("NEMOTRON_LLM_TEMPERATURE", DEFAULT_NEMOTRON_LLM_TEMPERATURE),
            max_tokens=_env_int("NEMOTRON_LLM_MAX_TOKENS", DEFAULT_NEMOTRON_LLM_MAX_TOKENS),
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}}},
        ),
    )

    # Text-to-Speech service
    tts = GradiumTTSService(
        api_key=os.environ["GRADIUM_API_KEY"],
        settings=GradiumTTSService.Settings(
            voice=os.getenv("GRADIUM_VOICE_ID", "Eu9iL_CYe8N-Gkx_"),
        ),
    )

    # ToolsSchema describes the tools to the LLM; register_pipecat_functions
    # wires the actual handlers the LLM will invoke. Both are required.
    register_pipecat_functions(llm)

    context = LLMContext(tools=tools)
    vad_analyzer = build_vad_analyzer(twilio=twilio, audio_in_sample_rate=audio_in_sample_rate)
    vad_params = vad_analyzer.params
    transport_name = "Twilio" if twilio else "WebRTC"
    logger.info(
        "{} VAD params confidence={} start_secs={} stop_secs={} min_volume={}",
        transport_name,
        vad_params.confidence,
        vad_params.start_secs,
        vad_params.stop_secs,
        vad_params.min_volume,
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=build_user_aggregator_params(vad_analyzer),
    )
    latency_logger = LatencyLogger(path=os.getenv("VOICE_LATENCY_LOG_PATH", "latency.jsonl"))
    stage_transcript_logger = StageLatencyLogger(
        path=os.getenv("VOICE_STAGE_LATENCY_LOG_PATH", "stage_latency.jsonl"),
        observe_upstream=True,
        observe_turn=True,
        observe_transcript=True,
        observe_llm=False,
        observe_audio=False,
        finalize=False,
    )
    stage_turn_logger = StageLatencyLogger(
        tracker=stage_transcript_logger.tracker,
        observe_turn=True,
        observe_transcript=False,
        observe_llm=True,
        observe_audio=False,
        finalize=False,
    )
    stage_audio_logger = StageLatencyLogger(
        tracker=stage_transcript_logger.tracker,
        observe_turn=False,
        observe_transcript=False,
        observe_llm=False,
        observe_audio=True,
        finalize=True,
    )
    front_desk_memory = MemoryFirstFrontDeskProcessor()

    # Pipeline - assembled from reusable components
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            stage_transcript_logger,
            user_aggregator,
            front_desk_memory,
            llm,
            stage_turn_logger,
            tts,
            transport.output(),
            latency_logger,
            stage_audio_logger,
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=audio_in_sample_rate,
            audio_out_sample_rate=audio_out_sample_rate,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation
        context.add_message(
            {
                "role": "user",
                "content": (
                    "A caller just reached the dental front desk. Greet them exactly: "
                    "'Thanks for calling Bright Smile Dental, this is Aria. How can I help?'"
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    from_number: str | None = None
    transport_overrides: dict = {}

    # Krisp is available when deployed to Pipecat Cloud
    if os.environ.get("ENV") != "local":
        from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter

        krisp_filter = KrispVivaFilter()
    else:
        krisp_filter = None

    match runner_args:
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection

            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_in_filter=krisp_filter,
                    audio_out_enabled=True,
                ),
            )
        case WebSocketRunnerArguments():
            # Twilio media streams arrive as 8 kHz μ-law, but the serializer
            # resamples input to the 16 kHz PCM contract used by VAD/STT.
            transport_overrides.update(twilio_transport_overrides())

            # Parse Twilio websocket and fetch call information
            _, call_data = await parse_telephony_websocket(runner_args.websocket)

            # Fetch call information from Twilio REST API for logging only.
            # Do not infer patient identity from caller ID.
            call_info = await get_call_info(call_data["call_id"])
            if call_info:
                from_number = call_info.get("from_number")
                logger.info(f"Call from: {from_number} to: {call_info.get('to_number')}")

            serializer = build_twilio_serializer(call_data)

            transport = FastAPIWebsocketTransport(
                websocket=runner_args.websocket,
                params=FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_in_filter=krisp_filter,
                    audio_out_enabled=True,
                    add_wav_header=False,
                    serializer=serializer,
                ),
            )
        case _:
            logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
            return

    await run_bot(
        transport,
        from_number=from_number,
        twilio=isinstance(runner_args, WebSocketRunnerArguments),
        **transport_overrides,
    )


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
