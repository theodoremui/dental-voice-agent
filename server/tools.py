"""Dental front-desk tools and prompt for the voice agent."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

OFFICE_HOURS = "Monday through Friday, 8:00 AM to 5:00 PM"
ACCEPTED_INSURANCE = {"delta dental", "metlife", "aetna"}

SYSTEM_PROMPT = f"""You are Aria, the front-desk assistant for Bright Smile Dental.
You can book, reschedule, and cancel appointments; answer office hours and basic insurance questions.

Office hours: {OFFICE_HOURS}.
Accepted insurance known list: Delta Dental, MetLife, Aetna.

Rules:
- You are on a phone call. Keep replies to 1-2 short sentences.
- Ask one question at a time.
- Never give medical or dental advice. For clinical questions, offer to book a visit.
- If the caller describes an emergency like severe pain, facial swelling, trauma, or bleeding, tell them to call 911 or go to the ER now. If they still need dental follow-up, offer an urgent appointment.
- Only confirm an appointment after calling book_appointment and receiving a confirmation id.
- Before book_appointment returns, do not say or imply that an appointment is booked or will be booked. Say you are checking or can help.
- Before reschedule_appointment returns a rescheduled status, do not say or imply that an appointment is rescheduled or will be rescheduled. If the tool returns not_found, ask the caller to check the confirmation id.
- Only confirm a cancellation after calling cancel_appointment and receiving a canceled status.
- Before cancel_appointment returns a canceled status, do not say or imply that an appointment is canceled or will be canceled. If the tool returns not_found, ask the caller to check the confirmation id.
- If a reschedule or cancellation lookup returns not_found, do not book a replacement appointment in the same flow. That can create duplicate appointments. Ask the caller to verify the confirmation id or call back with the correct id.
- Never invent insurance coverage. If a plan is not in the known list, say the office will confirm it.
- Say a short goodbye before calling end_call.
"""


def build_system_instruction(
    from_number: str | None = None,
    today: date | None = None,
) -> str:
    """Build the exact system instruction used by the voice bot and text evals."""
    today_value = today or date.today()
    caller_context = (
        f"Caller ID from Twilio is {from_number}. Do not read it aloud unless the caller asks."
        if from_number
        else "Caller ID is unavailable."
    )
    return (
        f"{SYSTEM_PROMPT.strip()}\n\n"
        "Voice behavior:\n"
        "- Sound like a capable dental front-desk coordinator, not a chatbot.\n"
        "- Keep each spoken turn brief. Longer is okay only for confirming appointment details.\n"
        "- Ask for one missing booking detail at a time: name, date, time, then reason.\n"
        "- For availability, use check_availability before offering appointment times.\n"
        "- For booking, rescheduling, cancellation, and insurance checks, use the matching tool.\n"
        "- Do not claim an appointment is booked, moved, or canceled until the tool confirms it.\n"
        "- If a tool returns not_found, ask the caller to check the confirmation id.\n"
        '- Skip filler like "Absolutely" or "Perfect" and go straight to the next useful step.\n'
        "- Responses are spoken aloud. No bullet points, markdown, or emojis.\n"
        "- When the caller is done or says goodbye, say a short closing line and call end_call "
        "in the same turn.\n\n"
        f"Today is {today_value.strftime('%A, %B %d, %Y')}. Use this when the caller gives "
        'a relative appointment date like "tomorrow" or "next Tuesday".\n\n'
        f"Caller context: {caller_context}"
    )


# ---- mock backend (in-memory) ----
_BOOKINGS: dict[str, dict[str, Any]] = {}
_next_id = [1000]


@dataclass(frozen=True)
class PipecatToolResult:
    value: dict[str, Any]
    properties: Any | None = None


def _check_availability(args: Mapping[str, Any], *, params: Any | None = None) -> dict[str, Any]:
    # Pretend afternoons are open, mornings are mostly full.
    date = str(args.get("date", "")).strip()
    return {"date": date, "open_slots": ["1:00 PM", "2:30 PM", "4:00 PM"]}


def _book_appointment(args: Mapping[str, Any], *, params: Any | None = None) -> dict[str, Any]:
    _next_id[0] += 1
    confirmation_id = f"BSD{_next_id[0]}"
    booking = {
        "name": args.get("name"),
        "date": args.get("date"),
        "time": args.get("time"),
        "reason": args.get("reason"),
    }
    _BOOKINGS[confirmation_id] = booking
    return {"confirmation_id": confirmation_id, "status": "booked", **booking}


def _reschedule_appointment(
    args: Mapping[str, Any], *, params: Any | None = None
) -> dict[str, Any]:
    confirmation_id = args.get("confirmation_id")
    if confirmation_id not in _BOOKINGS:
        return {"status": "not_found", "confirmation_id": confirmation_id}

    _BOOKINGS[confirmation_id].update({"date": args.get("date"), "time": args.get("time")})
    return {
        "status": "rescheduled",
        "confirmation_id": confirmation_id,
        **_BOOKINGS[confirmation_id],
    }


def _cancel_appointment(args: Mapping[str, Any], *, params: Any | None = None) -> dict[str, Any]:
    confirmation_id = args.get("confirmation_id")
    if confirmation_id not in _BOOKINGS:
        return {"status": "not_found", "confirmation_id": confirmation_id}

    canceled = _BOOKINGS.pop(str(confirmation_id))
    return {"status": "canceled", "confirmation_id": confirmation_id, **canceled}


def _check_insurance(args: Mapping[str, Any], *, params: Any | None = None) -> dict[str, Any]:
    provider = str(args.get("provider") or "").strip()
    return {
        "provider": provider,
        "accepted": provider.lower() in ACCEPTED_INSURANCE,
        "known_list_only": True,
    }


async def _end_call(args: Mapping[str, Any], *, params: Any | None = None) -> PipecatToolResult:
    if params is not None:
        from pipecat.frames.frames import EndTaskFrame, FunctionCallResultProperties
        from pipecat.processors.frame_processor import FrameDirection

        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        return PipecatToolResult(
            {"ok": True}, properties=FunctionCallResultProperties(run_llm=False)
        )

    return PipecatToolResult({"ok": True})


TOOL_IMPLS = {
    "check_availability": _check_availability,
    "book_appointment": _book_appointment,
    "reschedule_appointment": _reschedule_appointment,
    "cancel_appointment": _cancel_appointment,
    "check_insurance": _check_insurance,
    "end_call": _end_call,
}


# ---- OpenAI-style tool schemas (one source of truth for both code paths) ----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "List open appointment slots for a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Appointment date, e.g. 2026-06-02"}
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book an appointment. Call only after collecting name, date, time, and reason. "
                "Do not tell the caller the appointment is booked until this tool returns a "
                "confirmation_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "date", "time", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": (
                "Move an existing appointment to a new date and time. Call after the caller gives "
                "confirmation_id, date, and time. Do not tell the caller the appointment is "
                "rescheduled until this tool returns status=rescheduled; if it returns not_found, "
                "ask the caller to check the confirmation id. Do not book a replacement appointment "
                "as a workaround for not_found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmation_id": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                },
                "required": ["confirmation_id", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": (
                "Cancel an existing appointment by confirmation id. Do not tell the caller the "
                "appointment is canceled until this tool returns status=canceled; if it returns "
                "not_found, ask the caller to check the confirmation id. Do not book a replacement "
                "appointment as a workaround for not_found."
            ),
            "parameters": {
                "type": "object",
                "properties": {"confirmation_id": {"type": "string"}},
                "required": ["confirmation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_insurance",
            "description": "Check whether an insurance provider is in the accepted known list.",
            "parameters": {
                "type": "object",
                "properties": {"provider": {"type": "string"}},
                "required": ["provider"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the phone call. Call only after saying a short goodbye in the same turn.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def build_pipecat_tools_schema():
    """Convert the OpenAI-style tool declarations into Pipecat's ToolsSchema."""
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    standard_tools = []
    for tool in TOOLS:
        function = tool["function"]
        parameters = function.get("parameters", {})
        standard_tools.append(
            FunctionSchema(
                name=function["name"],
                description=function.get("description", ""),
                properties=parameters.get("properties", {}),
                required=parameters.get("required", []),
            )
        )

    return ToolsSchema(standard_tools=standard_tools)


def register_pipecat_functions(llm) -> None:
    """Register every tool implementation with a Pipecat LLM service."""

    def make_handler(name: str):
        async def handler(params):
            result = TOOL_IMPLS[name](params.arguments or {}, params=params)
            if inspect.isawaitable(result):
                result = await result

            if isinstance(result, PipecatToolResult):
                await params.result_callback(result.value, properties=result.properties)
            else:
                await params.result_callback(result)

        return handler

    for name in TOOL_IMPLS:
        llm.register_function(name, make_handler(name))
