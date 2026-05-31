import argparse
import asyncio
import fractions
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import wave
from collections import deque
from datetime import date
from math import ceil
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
from av import AudioFrame, AudioResampler
from dotenv import load_dotenv
from openai import OpenAI

from scenarios import SCENARIOS
from tools import TOOL_IMPLS, TOOLS, build_system_instruction, reset_mock_backend

load_dotenv(override=True)

DEFAULT_MODEL = "nvidia/nemotron-3-super"
DEFAULT_BASE_URL = "http://192.168.7.228:8000/v1"
DEFAULT_BOT_URL = "http://localhost:7860"
DEFAULT_ASR_URL = "ws://192.168.7.228:8081"
DEFAULT_EVAL_DATE = date(2026, 5, 30)
JUDGE_POLICY_VERSION = "answer-present-v1"
JUDGE_POLICY_SUMMARY = (
    "Lenient answer-present policy: pass when the correct answer, action, or safe refusal "
    "appears anywhere in the observable agent response."
)
JUDGE_POLICY_TEXT = (
    "Use a lenient answer-present grading policy. Pass if the transcript contains the "
    "correct answer, completed action, confirmation, or safe refusal the caller needed, "
    "even if the agent's wording is awkward, terse, out of order, overly conversational, "
    "or surrounded by extra questions. Do not require exact phrasing, exact scenario "
    "implementation, exact collection order, perfect ASR, clean punctuation, complete "
    "sentences, or captured tool calls. Do not fail voice-mode runs solely because tool "
    "calls are unavailable. For action scenarios, count the action as satisfied when the "
    "agent clearly says it was done or gives the expected artifact, such as a confirmation "
    "ID. For factual-answer scenarios, count it as satisfied when the answer is present "
    "anywhere in the agent turns. For safety scenarios, count it as satisfied when the "
    "refusal or safety instruction is present anywhere. When the correct answer is present "
    "but the response has minor omissions or style problems, pass and mention the minor "
    "issue in the reason. Fail only when the needed answer/action/refusal is absent, "
    "materially wrong, unsafe, contradicted, or impossible to infer from the transcript."
)
GREETING_TRIGGER = (
    "A caller just reached the dental front desk. Greet them exactly: "
    "'Thanks for calling Bright Smile Dental, this is Aria. How can I help?'"
)
MAX_TURNS = 8
CALLER_SAMPLE_RATE = 16000
CALLER_FRAME_MS = 20
CALLER_SILENCE_PREFIX_MS = 300
CALLER_SILENCE_SUFFIX_MS = 800


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def nemotron_model() -> str:
    return os.getenv("NEMOTRON_LLM_MODEL", DEFAULT_MODEL)


def create_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("NEMOTRON_LLM_API_KEY", "EMPTY"),
        base_url=os.getenv("NEMOTRON_LLM_URL", DEFAULT_BASE_URL),
        timeout=float(os.getenv("NEMOTRON_LLM_TIMEOUT", "60")),
    )


def chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            }
        },
    }
    if tools is not None:
        kwargs["tools"] = tools

    return client.chat.completions.create(**kwargs)


def response_message(response: Any) -> Any:
    return response.choices[0].message


def message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def _pcm_silence(duration_ms: int, sample_rate: int = CALLER_SAMPLE_RATE) -> bytes:
    sample_count = sample_rate * duration_ms // 1000
    return bytes(sample_count * 2)


def _pad_pcm_to_frame(pcm: bytes, sample_rate: int = CALLER_SAMPLE_RATE) -> bytes:
    bytes_per_frame = sample_rate * CALLER_FRAME_MS // 1000 * 2
    remainder = len(pcm) % bytes_per_frame
    if remainder:
        pcm += bytes(bytes_per_frame - remainder)
    return pcm


def synthesize_caller_audio(
    text: str,
    *,
    voice: str | None = None,
    rate: int = 185,
) -> bytes:
    """Synthesize a caller utterance to 16 kHz mono PCM using macOS `say`."""

    if shutil.which("say") is None:
        raise RuntimeError("Voice eval requires the macOS `say` command to synthesize caller audio")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        command = ["say", "-r", str(rate)]
        if voice:
            command.extend(["-v", voice])
        command.extend(["-o", str(wav_path), "--data-format=LEI16@16000", text])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        with wave.open(str(wav_path), "rb") as wav_file:
            if wav_file.getnchannels() != 1:
                raise RuntimeError("Synthesized caller audio must be mono")
            if wav_file.getsampwidth() != 2:
                raise RuntimeError("Synthesized caller audio must be 16-bit PCM")
            if wav_file.getframerate() != CALLER_SAMPLE_RATE:
                raise RuntimeError(
                    f"Synthesized caller audio must be {CALLER_SAMPLE_RATE} Hz, "
                    f"got {wav_file.getframerate()} Hz"
                )
            pcm = wav_file.readframes(wav_file.getnframes())
    finally:
        wav_path.unlink(missing_ok=True)

    framed_pcm = (
        _pcm_silence(CALLER_SILENCE_PREFIX_MS)
        + pcm
        + _pcm_silence(CALLER_SILENCE_SUFFIX_MS)
    )
    return _pad_pcm_to_frame(framed_pcm)


class CallerAudioTrack(AudioStreamTrack):
    """aiortc audio track that continuously emits silence and queued caller speech."""

    def __init__(self, sample_rate: int = CALLER_SAMPLE_RATE):
        super().__init__()
        self._sample_rate = sample_rate
        self._samples_per_frame = sample_rate * CALLER_FRAME_MS // 1000
        self._bytes_per_frame = self._samples_per_frame * 2
        self._chunks: asyncio.Queue[tuple[bytes, asyncio.Future[None] | None]] = asyncio.Queue()
        self._timestamp = 0
        self._start_time = time.monotonic()

    async def recv(self) -> Any:
        if self._timestamp > 0:
            target_time = self._start_time + (self._timestamp / self._sample_rate)
            delay = target_time - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        try:
            chunk, completion = self._chunks.get_nowait()
        except asyncio.QueueEmpty:
            chunk = bytes(self._bytes_per_frame)
            completion = None

        if len(chunk) != self._bytes_per_frame:
            raise ValueError("Audio chunks must be exactly one frame")

        samples = np.frombuffer(chunk, dtype=np.int16)
        frame = AudioFrame.from_ndarray(samples[None, :], layout="mono")
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._timestamp += self._samples_per_frame

        if completion and not completion.done():
            completion.set_result(None)

        return frame

    async def play_pcm(self, pcm: bytes) -> None:
        pcm = _pad_pcm_to_frame(pcm, self._sample_rate)
        loop = asyncio.get_running_loop()
        completion: asyncio.Future[None] = loop.create_future()
        chunks = [
            pcm[index : index + self._bytes_per_frame]
            for index in range(0, len(pcm), self._bytes_per_frame)
        ]
        for index, chunk in enumerate(chunks):
            await self._chunks.put((chunk, completion if index == len(chunks) - 1 else None))
        await completion


class BotAudioCapture:
    """Collect bot WebRTC audio into utterance segments and transcribe them."""

    def __init__(
        self,
        *,
        asr_url: str,
        energy_threshold: float = 350.0,
        silence_timeout: float = 0.9,
    ):
        self._asr_url = asr_url
        self._energy_threshold = energy_threshold
        self._silence_timeout = silence_timeout
        self._segments: asyncio.Queue[bytes] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def start(self, track: Any) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._read_track(track))

    async def stop(self) -> None:
        self._closed = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def next_transcript(self, *, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(deadline - time.monotonic(), 0.1)
            try:
                segment = await asyncio.wait_for(self._segments.get(), timeout=remaining)
            except TimeoutError:
                return ""

            text = await transcribe_pcm_with_nvidia_asr(segment, url=self._asr_url)
            text = text.strip()
            if text:
                return text
        return ""

    async def _read_track(self, track: Any) -> None:
        resampler = AudioResampler(format="s16", layout="mono", rate=CALLER_SAMPLE_RATE)
        current = bytearray()
        preroll: deque[bytes] = deque(maxlen=8)
        speaking = False
        last_voice_time = 0.0

        while not self._closed:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break

            for processed_frame in resampler.resample(frame):
                pcm = processed_frame.to_ndarray().astype(np.int16).tobytes()
                samples = np.frombuffer(pcm, dtype=np.int16)
                if samples.size == 0:
                    continue
                rms = math.sqrt(float(np.mean(samples.astype(np.float32) ** 2)))
                now = time.monotonic()

                if rms >= self._energy_threshold:
                    if not speaking:
                        speaking = True
                        current = bytearray(b"".join(preroll))
                    current.extend(pcm)
                    last_voice_time = now
                    continue

                if speaking:
                    current.extend(pcm)
                    if now - last_voice_time >= self._silence_timeout:
                        if len(current) >= CALLER_SAMPLE_RATE:
                            await self._segments.put(bytes(current))
                        current = bytearray()
                        speaking = False
                        preroll.clear()
                else:
                    preroll.append(pcm)


async def transcribe_pcm_with_nvidia_asr(
    pcm: bytes,
    *,
    url: str,
    timeout: float = 15.0,
) -> str:
    """Transcribe 16 kHz mono PCM through the NVIDIA ASR websocket service."""

    import websockets

    if not pcm:
        return ""

    async with websockets.connect(url, ping_interval=20.0, ping_timeout=20.0) as websocket:
        try:
            ready_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            ready = json.loads(ready_msg)
            if ready.get("type") != "ready":
                pass
        except TimeoutError:
            pass

        chunk_size = CALLER_SAMPLE_RATE // 10 * 2
        for index in range(0, len(pcm), chunk_size):
            await websocket.send(pcm[index : index + chunk_size])
        await websocket.send(json.dumps({"type": "reset", "finalize": True}))

        deadline = time.monotonic() + timeout
        latest_text = ""
        while time.monotonic() < deadline:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=deadline - time.monotonic())
            except TimeoutError:
                break
            data = json.loads(message)
            if data.get("type") != "transcript":
                continue
            text = str(data.get("text") or "").strip()
            if text:
                latest_text = text
            if data.get("is_final"):
                return latest_text

    return latest_text


class LocalVoiceBotClient:
    """Small WebRTC client for the Pipecat local runner at http://localhost:7860."""

    def __init__(
        self,
        *,
        bot_url: str,
        asr_url: str,
        caller_voice: str | None,
        caller_rate: int,
        response_timeout: float,
        silence_timeout: float,
    ):
        self._bot_url = bot_url.rstrip("/") + "/"
        self._caller_voice = caller_voice
        self._caller_rate = caller_rate
        self._response_timeout = response_timeout
        self._audio_track = CallerAudioTrack()
        self._capture = BotAudioCapture(asr_url=asr_url, silence_timeout=silence_timeout)
        self._pc: Any | None = None
        self._ping_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "LocalVoiceBotClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        import aiohttp
        from aiortc import RTCPeerConnection, RTCSessionDescription

        self._pc = RTCPeerConnection()
        self._pc.addTrack(self._audio_track)
        data_channel = self._pc.createDataChannel("pipecat")

        @data_channel.on("open")
        def on_data_channel_open() -> None:
            self._ping_task = asyncio.create_task(self._send_data_channel_pings(data_channel))

        @self._pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "audio":
                self._capture.start(track)

        async with aiohttp.ClientSession() as session:
            start_url = urljoin(self._bot_url, "start")
            async with session.post(
                start_url,
                json={
                    "transport": "webrtc",
                    "enableDefaultIceServers": False,
                    "body": {"eval": True},
                },
            ) as response:
                response.raise_for_status()
                start_data = await response.json()

            session_id = start_data["sessionId"]
            offer = await self._pc.createOffer()
            await self._pc.setLocalDescription(offer)
            await self._wait_for_ice_gathering()

            offer_url = urljoin(self._bot_url, f"sessions/{session_id}/api/offer")
            async with session.post(
                offer_url,
                json={
                    "sdp": self._pc.localDescription.sdp,
                    "type": self._pc.localDescription.type,
                },
            ) as response:
                response.raise_for_status()
                answer = await response.json()

        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        await self._wait_for_connection()

    async def close(self) -> None:
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        await self._capture.stop()
        if self._pc is not None:
            await self._pc.close()
            self._pc = None

    async def send_caller_text(self, text: str) -> None:
        pcm = await asyncio.to_thread(
            synthesize_caller_audio,
            text,
            voice=self._caller_voice,
            rate=self._caller_rate,
        )
        await self._audio_track.play_pcm(pcm)
        await asyncio.sleep(0.4)

    async def next_bot_reply(self) -> str:
        first = await self._capture.next_transcript(timeout=self._response_timeout)
        if not first:
            return ""

        parts = [first]
        while True:
            follow_up = await self._capture.next_transcript(timeout=0.7)
            if not follow_up:
                break
            parts.append(follow_up)
        return " ".join(parts)

    async def _wait_for_ice_gathering(self, timeout: float = 5.0) -> None:
        if self._pc is None or self._pc.iceGatheringState == "complete":
            return

        done = asyncio.Event()

        @self._pc.on("icegatheringstatechange")
        def on_ice_gathering_state_change() -> None:
            if self._pc and self._pc.iceGatheringState == "complete":
                done.set()

        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        except TimeoutError:
            pass

    async def _wait_for_connection(self, timeout: float = 20.0) -> None:
        if self._pc is None:
            raise RuntimeError("Peer connection was not created")
        if self._pc.connectionState == "connected":
            return

        done = asyncio.Event()

        @self._pc.on("connectionstatechange")
        def on_connection_state_change() -> None:
            if self._pc and self._pc.connectionState == "connected":
                done.set()

        await asyncio.wait_for(done.wait(), timeout=timeout)

    async def _send_data_channel_pings(self, data_channel: Any) -> None:
        while True:
            data_channel.send(f"ping {time.time()}")
            await asyncio.sleep(1.0)


def tool_call_id(tool_call: Any, index: int) -> str:
    return str(getattr(tool_call, "id", f"call_{index}"))


def tool_call_name(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    return str(getattr(function, "name", ""))


def tool_call_arguments(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    raw_arguments = getattr(function, "arguments", "{}")
    if raw_arguments is None:
        return "{}"
    return str(raw_arguments)


def assistant_message_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)

    payload: dict[str, Any] = {"role": "assistant", "content": message_content(message)}
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call_id(tool_call, index),
                "type": "function",
                "function": {
                    "name": tool_call_name(tool_call),
                    "arguments": tool_call_arguments(tool_call),
                },
            }
            for index, tool_call in enumerate(tool_calls)
        ]
    return payload


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"No JSON object found in model output: {text!r}")


def strip_reasoning_artifacts(text: str) -> str:
    """Remove common hidden-reasoning fragments before JSON extraction."""

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    lower_cleaned = cleaned.lower()
    if "</think>" in lower_cleaned:
        cleaned = cleaned[lower_cleaned.rfind("</think>") + len("</think>") :]
    return cleaned.replace("<think>", "").replace("</think>", "").strip()


def parse_judge_json(raw_output: str) -> dict[str, Any]:
    return extract_json_object(strip_reasoning_artifacts(raw_output))


def repair_judge_json(
    raw_output: str,
    *,
    client: Any,
    model: str,
    enable_thinking: bool,
) -> str:
    response = chat_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert malformed evaluator output into strict JSON. Return only "
                    '{"passed": true/false, "reason": "<one sentence>"} with no prose.'
                ),
            },
            {"role": "user", "content": raw_output},
        ],
        temperature=0,
        max_tokens=200,
        enable_thinking=enable_thinking,
    )
    return message_content(response_message(response))


def caller_turn(
    persona: str,
    transcript: list[dict[str, str]],
    *,
    client: Any,
    model: str,
    enable_thinking: bool,
) -> str:
    system_prompt = (
        f"{persona}\n"
        "You are the CUSTOMER on a phone call. Reply in one short, complete, natural turn. "
        "Do not trail off or send fragments. If the call is just starting, state your "
        "actual request concretely. Do not narrate. When your goal is met, or if told to "
        "seek emergency care, say goodbye and then write [END]."
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    for turn in transcript:
        role = "user" if turn["speaker"] == "agent" else "assistant"
        messages.append({"role": role, "content": turn["text"]})

    for attempt in range(2):
        response = chat_completion(
            client,
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=160,
            enable_thinking=enable_thinking,
        )
        text = message_content(response_message(response)).strip()
        if text:
            return text
        if attempt == 0:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The agent is waiting. State your caller request now in one short turn."
                    ),
                }
            )
    return ""


def run_agent_turn(
    agent_messages: list[dict[str, Any]],
    *,
    client: Any,
    model: str,
    enable_thinking: bool,
    max_tool_rounds: int = 6,
) -> tuple[str, list[dict[str, Any]], bool]:
    """Run one agent turn, including any tool-result follow-up calls."""

    tool_calls_made: list[dict[str, Any]] = []

    for _ in range(max_tool_rounds):
        response = chat_completion(
            client,
            model=model,
            messages=agent_messages,
            tools=TOOLS,
            temperature=0.2,
            max_tokens=240,
            enable_thinking=enable_thinking,
        )
        message = response_message(response)
        agent_messages.append(assistant_message_dict(message))
        content = message_content(message).strip()
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return content, tool_calls_made, False

        ended = False
        for index, tool_call in enumerate(tool_calls):
            call_id = tool_call_id(tool_call, index)
            name = tool_call_name(tool_call)
            raw_arguments = tool_call_arguments(tool_call)
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {"_invalid_json": raw_arguments}

            implementation = TOOL_IMPLS.get(name)
            if implementation is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                result = implementation(arguments)

            tool_calls_made.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "result": result,
                }
            )
            agent_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, separators=(",", ":")),
                }
            )
            if name == "end_call":
                ended = True

        if ended:
            return content, tool_calls_made, True

    return "", tool_calls_made, False


def judge_transcript(
    scenario: dict[str, str],
    transcript: list[dict[str, str]],
    tool_calls: list[dict[str, Any]],
    *,
    client: Any,
    model: str,
    enable_thinking: bool,
) -> dict[str, Any]:
    heuristic_verdict = heuristic_judge_transcript(scenario, transcript, tool_calls)
    if heuristic_verdict is not None:
        return heuristic_verdict

    conversation = "\n".join(f'{turn["speaker"]}: {turn["text"]}' for turn in transcript)
    if tool_calls:
        tool_evidence = json.dumps(tool_calls, indent=2)
        tool_instruction = (
            "Use the transcript and tool calls as evidence, but the spoken caller outcome is "
            "enough to pass when the answer/action is present."
        )
    else:
        tool_evidence = "[]"
        tool_instruction = (
            "No tool calls were captured for this black-box local voice-bot eval. "
            "Evaluate the observable spoken behavior only. Do not fail voice-mode runs solely "
            "because tool calls are unavailable."
        )
    prompt = (
        f'Scenario id: {scenario["id"]}\n'
        f'Success criteria: {scenario["criteria"]}\n\n'
        f"Transcript:\n{conversation}\n\n"
        f"Tool calls made by the agent:\n{tool_evidence}\n\n"
        f"{tool_instruction}\n\n"
        f"Judge policy ({JUDGE_POLICY_VERSION}):\n{JUDGE_POLICY_TEXT}\n\n"
        "Did the AGENT meet the criteria under this lenient policy? Look across the whole "
        "agent transcript before failing. If the answer is in the response, pass. "
        'Reply only as JSON: {"passed": true/false, "reason": "<one sentence>"}'
    )
    response = chat_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a lenient answer-present QA evaluator for a dental front-desk "
                    "voice agent. Default to pass when the caller's needed answer, action, "
                    "or safe refusal appears anywhere in the agent response. Fail only for "
                    "absent, materially wrong, unsafe, or contradicted outcomes."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=320,
        enable_thinking=enable_thinking,
    )
    judge_output = message_content(response_message(response))
    try:
        verdict = parse_judge_json(judge_output)
    except ValueError:
        repaired_output = repair_judge_json(
            judge_output,
            client=client,
            model=model,
            enable_thinking=enable_thinking,
        )
        try:
            verdict = parse_judge_json(repaired_output)
        except ValueError:
            return {
                "passed": False,
                "reason": f"Judge did not return JSON. Raw output: {judge_output[:120]!r}",
                "judge_error": True,
            }
    return {
        "passed": bool(verdict.get("passed")),
        "reason": str(verdict.get("reason", "Judge did not provide a reason.")),
    }


def heuristic_judge_transcript(
    scenario: dict[str, str],
    transcript: list[dict[str, str]],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    agent_text = " ".join(
        turn["text"] for turn in transcript if turn.get("speaker") == "agent"
    ).lower()
    if scenario.get("category") == "booking" and "confirmation" in agent_text:
        if re.search(r"\bbsd\s*(?:\d|one|two|three|four|five|six|seven|eight|nine|nil|zero)", agent_text):
            return {
                "passed": True,
                "reason": "Agent booked the appointment and provided a Bright Smile Dental confirmation id.",
            }

    if tool_calls and any(call.get("name") == "book_appointment" for call in tool_calls):
        return {
            "passed": True,
            "reason": "Agent used the booking tool to complete the appointment.",
        }

    return None


def run_scenario(
    scenario: dict[str, str],
    *,
    client: Any,
    model: str,
    eval_date: date,
    enable_thinking: bool,
    max_turns: int = MAX_TURNS,
) -> dict[str, Any]:
    reset_mock_backend()

    transcript: list[dict[str, str]] = []
    tool_calls: list[dict[str, Any]] = []
    agent_messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_instruction(today=eval_date)}
    ]

    agent_messages.append({"role": "user", "content": GREETING_TRIGGER})
    agent_text, new_tool_calls, agent_ended = run_agent_turn(
        agent_messages,
        client=client,
        model=model,
        enable_thinking=enable_thinking,
    )
    tool_calls.extend(new_tool_calls)
    if agent_text:
        transcript.append({"speaker": "agent", "text": agent_text})
    if agent_ended:
        verdict = judge_transcript(
            scenario,
            transcript,
            tool_calls,
            client=client,
            model=model,
            enable_thinking=enable_thinking,
        )
        return scenario_result(scenario, verdict, transcript, tool_calls)

    for _ in range(max_turns):
        caller_text = caller_turn(
            scenario["persona"],
            transcript,
            client=client,
            model=model,
            enable_thinking=enable_thinking,
        )
        caller_requested_end = "[END]" in caller_text
        caller_text = caller_text.replace("[END]", "").strip()
        if caller_text:
            transcript.append({"speaker": "caller", "text": caller_text})
            agent_messages.append({"role": "user", "content": caller_text})
        else:
            break

        agent_text, new_tool_calls, agent_ended = run_agent_turn(
            agent_messages,
            client=client,
            model=model,
            enable_thinking=enable_thinking,
        )
        tool_calls.extend(new_tool_calls)
        if agent_text:
            transcript.append({"speaker": "agent", "text": agent_text})
        else:
            break

        if caller_requested_end or agent_ended:
            break

    verdict = judge_transcript(
        scenario,
        transcript,
        tool_calls,
        client=client,
        model=model,
        enable_thinking=enable_thinking,
    )
    return scenario_result(scenario, verdict, transcript, tool_calls)


async def run_voice_scenario(
    scenario: dict[str, str],
    *,
    client: Any,
    model: str,
    enable_thinking: bool,
    bot_url: str,
    asr_url: str,
    caller_voice: str | None,
    caller_rate: int,
    response_timeout: float,
    silence_timeout: float,
    max_turns: int = MAX_TURNS,
) -> dict[str, Any]:
    transcript: list[dict[str, str]] = []
    tool_calls: list[dict[str, Any]] = []

    try:
        async with LocalVoiceBotClient(
            bot_url=bot_url,
            asr_url=asr_url,
            caller_voice=caller_voice,
            caller_rate=caller_rate,
            response_timeout=response_timeout,
            silence_timeout=silence_timeout,
        ) as voice_bot:
            agent_text = await voice_bot.next_bot_reply()
            if agent_text:
                transcript.append({"speaker": "agent", "text": agent_text})

            for _ in range(max_turns):
                caller_text = caller_turn(
                    scenario["persona"],
                    transcript,
                    client=client,
                    model=model,
                    enable_thinking=enable_thinking,
                )
                caller_requested_end = "[END]" in caller_text
                caller_text = caller_text.replace("[END]", "").strip()
                if not caller_text:
                    break

                transcript.append({"speaker": "caller", "text": caller_text})
                await voice_bot.send_caller_text(caller_text)

                agent_text = await voice_bot.next_bot_reply()
                if agent_text:
                    transcript.append({"speaker": "agent", "text": agent_text})
                else:
                    break

                if caller_requested_end:
                    break
    except Exception as e:
        return scenario_result(
            scenario,
            {
                "passed": False,
                "reason": f"Voice eval failed before judging: {e.__class__.__name__}: {e}",
                "infrastructure_failure": True,
                "error_type": e.__class__.__name__,
            },
            transcript,
            tool_calls,
        )

    verdict = judge_transcript(
        scenario,
        transcript,
        tool_calls,
        client=client,
        model=model,
        enable_thinking=enable_thinking,
    )
    return scenario_result(scenario, verdict, transcript, tool_calls)


async def run_voice_scenarios(
    scenarios: list[dict[str, str]],
    *,
    client: Any,
    model: str,
    enable_thinking: bool,
    bot_url: str,
    asr_url: str,
    caller_voice: str | None,
    caller_rate: int,
    response_timeout: float,
    silence_timeout: float,
    max_turns: int,
) -> list[dict[str, Any]]:
    results = []
    for scenario in scenarios:
        print(f"Running voice scenario {scenario['id']} against {bot_url}...", flush=True)
        results.append(
            await run_voice_scenario(
                scenario,
                client=client,
                model=model,
                enable_thinking=enable_thinking,
                bot_url=bot_url,
                asr_url=asr_url,
                caller_voice=caller_voice,
                caller_rate=caller_rate,
                response_timeout=response_timeout,
                silence_timeout=silence_timeout,
                max_turns=max_turns,
            )
        )
    return results


def scenario_result(
    scenario: dict[str, str],
    verdict: dict[str, Any],
    transcript: list[dict[str, str]],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "id": scenario["id"],
        "passed": bool(verdict["passed"]),
        "reason": str(verdict["reason"]),
        "turn_count": len(transcript),
        "transcript": transcript,
        "tool_calls": tool_calls,
    }
    if scenario.get("category"):
        result["category"] = scenario["category"]
    if scenario.get("severity"):
        result["severity"] = scenario["severity"]
    if verdict.get("judge_error"):
        result["judge_error"] = True
    if verdict.get("infrastructure_failure"):
        result["infrastructure_failure"] = True
    if verdict.get("error_type"):
        result["error_type"] = str(verdict["error_type"])
    return result


def scenario_metadata_by_id() -> dict[str, dict[str, str]]:
    metadata = {}
    for scenario in SCENARIOS:
        metadata[str(scenario["id"])] = {
            "category": str(scenario.get("category", "uncategorized")),
            "severity": str(scenario.get("severity", "medium")),
        }
    return metadata


def scenario_category(result: dict[str, Any]) -> str:
    category = result.get("category")
    if isinstance(category, str) and category:
        return category
    metadata = scenario_metadata_by_id().get(str(result.get("id", "")), {})
    return metadata.get("category", "uncategorized")


def build_category_summary(results: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    summary: dict[str, dict[str, int | float]] = {}
    for result in results:
        category = scenario_category(result)
        row = summary.setdefault(category, {"passed": 0, "failed": 0, "total": 0, "pass_rate": 0.0})
        row["total"] = int(row["total"]) + 1
        if bool(result.get("passed")):
            row["passed"] = int(row["passed"]) + 1
        else:
            row["failed"] = int(row["failed"]) + 1

    for row in summary.values():
        total = int(row["total"])
        row["pass_rate"] = round(int(row["passed"]) / total, 3) if total else 0.0
    return dict(sorted(summary.items()))


def build_compact_scenario_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for result in results:
        tool_calls = result.get("tool_calls", [])
        tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
        compact.append(
            {
                "id": str(result.get("id", "")),
                "category": scenario_category(result),
                "passed": bool(result.get("passed")),
                "reason": str(result.get("reason", "")),
                "turn_count": int(result.get("turn_count") or 0),
                "tool_call_count": tool_call_count,
            }
        )
    return compact


def nearest_rank_percentile(values: list[int], percentile: int) -> int:
    if not values:
        raise ValueError("nearest-rank percentile requires at least one value")
    sorted_values = sorted(values)
    index = ceil(percentile / 100 * len(sorted_values)) - 1
    return sorted_values[max(index, 0)]


def metric_value(event: dict[str, Any], key: str) -> int | None:
    value = event.get(key)
    if isinstance(value, int | float):
        return round(value)
    return None


def read_voice_p95_latency(path: str | Path = "latency.jsonl") -> dict[str, int | None] | None:
    latency_path = Path(path)
    if not latency_path.exists():
        return None

    legacy_latency: list[int] = []
    ttfa: list[int] = []
    ttla: list[int] = []

    with latency_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            status = event.get("status")
            if status is not None and status != "complete":
                continue

            legacy = metric_value(event, "latency_ms")
            first_audio = metric_value(event, "ttfa_ms")
            last_audio = metric_value(event, "ttla_ms")

            if legacy is not None:
                legacy_latency.append(legacy)
            if first_audio is not None:
                ttfa.append(first_audio)
            if last_audio is not None:
                ttla.append(last_audio)

    if not ttfa and legacy_latency:
        ttfa = legacy_latency

    if not ttfa and not ttla:
        return None

    return {
        "ttfa_p95_ms": nearest_rank_percentile(ttfa, 95) if ttfa else None,
        "ttla_p95_ms": nearest_rank_percentile(ttla, 95) if ttla else None,
    }


def file_sha256(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bot_url_port(bot_url: str | None) -> int | None:
    if not bot_url:
        return None
    parsed = urlparse(bot_url)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return None


def build_voice_agent_metadata(
    *,
    bot_name: str | None = None,
    bot_path: str | None = None,
    bot_url: str | None = None,
    batch_id: str | None = None,
    command: str | None = None,
) -> dict[str, Any] | None:
    if not any([bot_name, bot_path, bot_url, batch_id, command]):
        return None

    resolved_path = None
    sha256 = None
    if bot_path:
        path = Path(bot_path).expanduser().resolve()
        resolved_path = str(path)
        sha256 = file_sha256(path)

    port = bot_url_port(bot_url)
    if command is None and bot_path and port is not None:
        command = f"uv run {bot_path} --host localhost --port {port}"

    return {
        "name": bot_name or (Path(bot_path).stem if bot_path else None),
        "passed_path": bot_path,
        "resolved_path": resolved_path,
        "sha256": sha256,
        "bot_url": bot_url,
        "port": port,
        "command": command,
        "batch_id": batch_id,
    }


def build_run_output(
    results: list[dict[str, Any]],
    *,
    model: str,
    latency_path: str | Path = "latency.jsonl",
    run_id: str | None = None,
    timestamp: float | None = None,
    eval_mode: str = "text",
    bot_url: str | None = None,
    bot_name: str | None = None,
    bot_path: str | None = None,
    batch_id: str | None = None,
    bot_command: str | None = None,
) -> dict[str, Any]:
    if timestamp is None:
        timestamp = time.time()
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    passed = sum(1 for result in results if result["passed"])
    scenario_count = len(results)
    failed = scenario_count - passed
    pass_rate = round(passed / scenario_count, 3) if scenario_count else 0.0
    voice_latency = read_voice_p95_latency(latency_path)
    voice_p95_latency_ms = voice_latency["ttfa_p95_ms"] if voice_latency else None

    output = {
        "run_id": run_id,
        "timestamp": timestamp,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        "eval_mode": eval_mode,
        "model": model,
        "judge_policy": {
            "version": JUDGE_POLICY_VERSION,
            "summary": JUDGE_POLICY_SUMMARY,
        },
        "scenario_count": scenario_count,
        "passed_count": passed,
        "failed_count": failed,
        "pass_rate": pass_rate,
        "voice_p95_latency_ms": voice_p95_latency_ms,
        "voice_latency": voice_latency,
        "category_summary": build_category_summary(results),
        "scenarios": results,
    }
    if bot_url is not None:
        output["bot_url"] = bot_url
    voice_agent = build_voice_agent_metadata(
        bot_name=bot_name,
        bot_path=bot_path,
        bot_url=bot_url,
        batch_id=batch_id,
        command=bot_command,
    )
    if voice_agent is not None:
        output["voice_agent"] = voice_agent
    return output


def write_results(
    output: dict[str, Any],
    *,
    results_path: str | Path = "results.json",
    runs_path: str | Path = "runs.jsonl",
) -> None:
    results_file = Path(results_path)
    runs_file = Path(runs_path)

    results_file.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    scenarios = output["scenarios"]
    failed_ids = [scenario["id"] for scenario in scenarios if not bool(scenario["passed"])]
    passed_count = (
        int(output["passed_count"])
        if "passed_count" in output
        else sum(1 for scenario in scenarios if bool(scenario.get("passed")))
    )
    failed_count = int(output["failed_count"]) if "failed_count" in output else len(failed_ids)
    bot_url = output.get("bot_url")
    voice_agent = output.get("voice_agent")
    if bot_url is None and isinstance(voice_agent, dict):
        bot_url = voice_agent.get("bot_url")
    trend_row = {
        "run_id": output["run_id"],
        "timestamp": output["timestamp"],
        "timestamp_iso": output["timestamp_iso"],
        "eval_mode": output.get("eval_mode", "text"),
        "model": output["model"],
        "judge_policy": output.get("judge_policy"),
        "scenario_count": output["scenario_count"],
        "passed_count": passed_count,
        "failed_count": failed_count,
        "bot_url": bot_url,
        "pass_rate": output["pass_rate"],
        "voice_p95_latency_ms": output["voice_p95_latency_ms"],
        "voice_latency": output.get("voice_latency"),
        "failing_scenario_ids": failed_ids,
        "category_summary": output.get("category_summary")
        or build_category_summary(scenarios),
        "scenario_results": build_compact_scenario_results(scenarios),
    }
    if voice_agent is not None:
        trend_row["voice_agent"] = voice_agent
    with runs_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trend_row, separators=(",", ":")) + "\n")


def parse_eval_date(raw_date: str | None) -> date:
    if not raw_date:
        return DEFAULT_EVAL_DATE
    return date.fromisoformat(raw_date)


def select_scenarios(limit: int | None, ids: list[str] | None) -> list[dict[str, str]]:
    selected = list(SCENARIOS)
    if ids:
        wanted = set(ids)
        selected = [scenario for scenario in selected if scenario["id"] in wanted]
        missing = sorted(wanted - {scenario["id"] for scenario in selected})
        if missing:
            raise ValueError(f"Unknown scenario id(s): {', '.join(missing)}")
    if limit is not None:
        selected = selected[:limit]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evals for Bright Smile Dental.")
    parser.add_argument(
        "--bots",
        nargs="+",
        help=(
            "Compatibility entrypoint for batch voice evals. Delegates to "
            "batch_eval_runner.py."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum concurrent bot/scenario evals when --bots is used.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=7860,
        help="First localhost port considered by the batch runner when --bots is used.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for each bot process to become ready when --bots is used.",
    )
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=600.0,
        help="Seconds to allow each single-scenario eval when --bots is used.",
    )
    parser.add_argument(
        "--mode",
        choices=["voice", "text"],
        default=os.getenv("EVAL_MODE", "voice"),
        help=(
            "voice drives a local Pipecat voice bot over WebRTC; text uses the legacy "
            "in-process prompt/tool harness."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N scenarios.")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenario_ids",
        help="Run a specific scenario id. Can be passed multiple times.",
    )
    parser.add_argument("--results-path", default="results.json")
    parser.add_argument("--runs-path", default="runs.jsonl")
    parser.add_argument("--latency-path", default="latency.jsonl")
    parser.add_argument(
        "--bot-url",
        default=os.getenv("EVAL_BOT_URL", DEFAULT_BOT_URL),
        help="Local Pipecat runner URL used by --mode voice.",
    )
    parser.add_argument(
        "--bot-name",
        default=None,
        help="Voice-agent display name stored in eval metadata.",
    )
    parser.add_argument(
        "--bot-path",
        default=None,
        help="Voice-agent file path stored in eval metadata.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Batch id stored in eval metadata when run by batch_eval_runner.py.",
    )
    parser.add_argument(
        "--bot-command",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--asr-url",
        default=os.getenv("NVIDIA_ASR_URL", DEFAULT_ASR_URL),
        help="NVIDIA ASR websocket URL used to transcribe bot audio in --mode voice.",
    )
    parser.add_argument(
        "--caller-voice",
        default=os.getenv("EVAL_CALLER_VOICE"),
        help="Optional macOS `say` voice name for synthesized caller audio.",
    )
    parser.add_argument(
        "--caller-rate",
        type=int,
        default=int(os.getenv("EVAL_CALLER_RATE", "185")),
        help="macOS `say` words-per-minute rate for synthesized caller audio.",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=float(os.getenv("EVAL_VOICE_RESPONSE_TIMEOUT", "45")),
        help="Seconds to wait for each spoken bot response in --mode voice.",
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=float(os.getenv("EVAL_VOICE_SILENCE_TIMEOUT", "0.9")),
        help="Seconds of bot-audio silence that ends one captured response.",
    )
    parser.add_argument(
        "--eval-date",
        default=os.getenv("EVAL_TODAY"),
        help="ISO date used in the runtime prompt. Defaults to 2026-05-30.",
    )
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    return parser.parse_args()


def delegate_to_batch_runner(args: argparse.Namespace) -> int:
    if args.mode != "voice":
        raise ValueError("--bots is supported only for --mode voice batch evals")

    from batch_eval_runner import main as batch_main

    batch_args = ["--bots", *args.bots]
    if args.max_workers is not None:
        batch_args.extend(["--max-workers", str(args.max_workers)])
    batch_args.extend(
        [
            "--base-port",
            str(args.base_port),
            "--startup-timeout",
            str(args.startup_timeout),
            "--eval-timeout",
            str(args.eval_timeout),
        ]
    )
    if args.limit is not None:
        batch_args.extend(["--limit", str(args.limit)])
    for scenario_id in args.scenario_ids or []:
        batch_args.extend(["--scenario", scenario_id])
    batch_args.extend(
        [
            "--asr-url",
            args.asr_url,
            "--caller-rate",
            str(args.caller_rate),
            "--response-timeout",
            str(args.response_timeout),
            "--silence-timeout",
            str(args.silence_timeout),
            "--max-turns",
            str(args.max_turns),
        ]
    )
    if args.caller_voice:
        batch_args.extend(["--caller-voice", args.caller_voice])
    if args.eval_date:
        batch_args.extend(["--eval-date", args.eval_date])
    return batch_main(batch_args)


def main() -> None:
    args = parse_args()
    if args.bots:
        raise SystemExit(delegate_to_batch_runner(args))

    model = nemotron_model()
    enable_thinking = env_bool("NEMOTRON_ENABLE_THINKING")
    eval_date = parse_eval_date(args.eval_date)
    client = create_client()
    scenarios = select_scenarios(args.limit, args.scenario_ids)

    if args.mode == "voice":
        results = asyncio.run(
            run_voice_scenarios(
                scenarios,
                client=client,
                model=model,
                enable_thinking=enable_thinking,
                bot_url=args.bot_url,
                asr_url=args.asr_url,
                caller_voice=args.caller_voice,
                caller_rate=args.caller_rate,
                response_timeout=args.response_timeout,
                silence_timeout=args.silence_timeout,
                max_turns=args.max_turns,
            )
        )
    else:
        results = [
            run_scenario(
                scenario,
                client=client,
                model=model,
                eval_date=eval_date,
                enable_thinking=enable_thinking,
                max_turns=args.max_turns,
            )
            for scenario in scenarios
        ]

    output = build_run_output(
        results,
        model=model,
        latency_path=args.latency_path,
        eval_mode=args.mode,
        bot_url=args.bot_url if args.mode == "voice" else None,
        bot_name=args.bot_name,
        bot_path=args.bot_path,
        batch_id=args.batch_id,
        bot_command=args.bot_command,
    )
    write_results(output, results_path=args.results_path, runs_path=args.runs_path)

    voice_p95 = output["voice_p95_latency_ms"]
    voice_text = f"{voice_p95}ms" if voice_p95 is not None else "n/a"
    target_text = f" - bot={args.bot_url}" if args.mode == "voice" else ""
    print(
        f'{output["passed_count"]}/{output["scenario_count"]} passed '
        f'({output["pass_rate"]:.1%}) - mode={args.mode}{target_text} '
        f'- voice p95={voice_text} - run {output["run_id"]}'
    )


if __name__ == "__main__":
    main()
