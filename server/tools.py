from datetime import date, datetime
from typing import Any

SYSTEM_PROMPT = """You are Aria, the front-desk assistant for Bright Smile Dental.
You can check appointment availability, book appointments, reschedule appointments, and
check whether Delta Dental, MetLife, or Aetna are on the accepted insurance list.
Rules:
- You are on a phone call. Keep replies to 1-2 short sentences.
- Ask one thing at a time.
- Do not claim you can cancel appointments.
- Do not confirm office hours; say the office will confirm current hours if asked.
- Never give medical, dental, diagnosis, treatment, or medication advice. For clinical
  questions, offer to book a visit.
- If the caller describes an emergency such as severe pain, facial swelling, trauma, or
  bleeding, tell them to seek emergency care first, then offer an urgent appointment slot.
- Only confirm an appointment after calling book_appointment and receiving a
  confirmation id.
- When the caller gives a date or relative date plus a vague time of day like
  afternoon, call check_availability for that date before asking for a specific
  time, then offer the open slots.
- If the caller already gave name, date, time, and reason, use the booking tools
  instead of asking for the same details again.
- If a requested time is not open, offer the closest open slot in one short
  sentence.
- Never invent insurance coverage. If a plan is not in your known list, say the office
  will confirm it.
- If the caller says goodbye or the call is complete, say a short goodbye and call
  end_call in the same turn.
Accepted insurance known list: Delta Dental, MetLife, Aetna.
"""


def build_system_instruction(
    today: date | None = None,
    from_number: str | None = None,
) -> str:
    """Build the exact runtime instruction used by the voice bot and text evals."""

    if today is None:
        today = date.today()

    caller_context = (
        "Twilio supplied caller ID for logging only. Do not infer patient identity, claim "
        "to recognize the caller, or mention caller records based on phone number alone."
        if from_number
        else "No caller ID context is available. Treat this as a standard front-desk call."
    )

    return (
        f"{SYSTEM_PROMPT}\n"
        "Phone style:\n"
        "- Talk like a concise dental front-desk staff member, not a chatbot.\n"
        "- Skip filler openers and do not restate what the caller just said.\n"
        "- Responses are spoken aloud. No bullet points and no emojis.\n"
        '- Say appointment times naturally, like "two thirty PM", not "two point thirty".\n\n'
        f"Today is {today.strftime('%A, %B %d, %Y')}. Use this when the caller "
        'gives a relative appointment date like "this Friday" or "next Tuesday".\n\n'
        f"Caller context: {caller_context}"
    )

# ---- mock backend (in-memory) ----
_DEMO_BOOKINGS: dict[str, dict[str, Any]] = {
    "BSD1001": {
        "name": "Demo Patient",
        "date": "2026-06-01",
        "time": "1:00 PM",
        "reason": "cleaning",
    }
}
_BOOKINGS: dict[str, dict[str, Any]] = {}
_next_id = [1001]


def reset_mock_backend(seed_demo_booking: bool = True) -> None:
    """Reset the in-memory backend so eval scenarios can run independently."""

    _BOOKINGS.clear()
    if seed_demo_booking:
        _BOOKINGS.update({key: value.copy() for key, value in _DEMO_BOOKINGS.items()})
        _next_id[0] = 1001
    else:
        _next_id[0] = 1000


reset_mock_backend()


def _check_availability(args: dict[str, Any]) -> dict[str, Any]:
    # Pretend afternoons are open, mornings are mostly full.
    appointment_date = args.get("date", "")
    return {"date": appointment_date, "open_slots": ["1:00 PM", "2:00 PM", "2:30 PM", "4:00 PM"]}


def _book_appointment(args: dict[str, Any]) -> dict[str, Any]:
    _next_id[0] += 1
    confirmation_id = f"BSD{_next_id[0]}"
    _BOOKINGS[confirmation_id] = args.copy()
    return {"confirmation_id": confirmation_id, "status": "booked", **args}


def _reschedule_appointment(args: dict[str, Any]) -> dict[str, Any]:
    confirmation_id = args.get("confirmation_id")
    if not isinstance(confirmation_id, str) or confirmation_id not in _BOOKINGS:
        return {"status": "not_found", "confirmation_id": confirmation_id}
    _BOOKINGS[confirmation_id].update(args)
    return {"status": "rescheduled", "confirmation_id": confirmation_id, **args}


def _check_insurance(args: dict[str, Any]) -> dict[str, Any]:
    known = {"delta dental", "metlife", "aetna"}
    provider = (args.get("provider") or "").strip().lower()
    return {
        "provider": args.get("provider"),
        "accepted": provider in known,
        "known_list_only": True,
    }


def _end_call_text_eval(args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "ended": True}


TOOL_IMPLS = {
    "check_availability": _check_availability,
    "book_appointment": _book_appointment,
    "reschedule_appointment": _reschedule_appointment,
    "check_insurance": _check_insurance,
    "end_call": _end_call_text_eval,
}

# ---- OpenAI-style tool schemas (one source of truth for both code paths) ----
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "List open dental appointment slots for a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Requested appointment date, e.g. 2026-06-02.",
                    }
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
                "Book a dental appointment. Call only after collecting name, date, time, "
                "and reason."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Patient's name."},
                    "date": {"type": "string", "description": "Appointment date."},
                    "time": {"type": "string", "description": "Appointment time."},
                    "reason": {"type": "string", "description": "Reason for the visit."},
                },
                "required": ["name", "date", "time", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Move an existing dental appointment to a new date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmation_id": {
                        "type": "string",
                        "description": "Existing Bright Smile Dental confirmation id.",
                    },
                    "date": {"type": "string", "description": "New appointment date."},
                    "time": {"type": "string", "description": "New appointment time."},
                },
                "required": ["confirmation_id", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_insurance",
            "description": "Check whether an insurance provider is in the accepted list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "description": "Insurance provider name."}
                },
                "required": ["provider"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": (
                "End the call. Only call this after saying goodbye to the caller in the "
                "same assistant turn."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _function_schema_from_openai_tool(tool: dict[str, Any]) -> Any:
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    function = tool["function"]
    parameters = function.get("parameters", {})
    return FunctionSchema(
        name=function["name"],
        description=function.get("description", ""),
        properties=parameters.get("properties", {}),
        required=parameters.get("required", []),
    )


def pipecat_tools_schema() -> Any:
    """Return Pipecat tool schemas derived from the OpenAI-style tool definitions."""
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    return ToolsSchema(standard_tools=[_function_schema_from_openai_tool(tool) for tool in TOOLS])


def get_pipecat_tools_schema() -> Any:
    """Compatibility helper with a more explicit name."""
    return pipecat_tools_schema()


async def _end_call(params: Any) -> None:
    from loguru import logger
    from pipecat.frames.frames import EndTaskFrame, FunctionCallResultProperties
    from pipecat.processors.frame_processor import FrameDirection

    logger.info("end_call invoked - pushing EndTaskFrame upstream")
    await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
    await params.result_callback(
        {"ok": True},
        properties=FunctionCallResultProperties(run_llm=False),
    )


def _human_date(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "that date"
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
        return parsed.strftime("%A, %B %d").replace(" 0", " ")
    except ValueError:
        return value


def _spoken_time(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        parsed = datetime.strptime(value.strip(), "%I:%M %p")
    except ValueError:
        return value

    hour_words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
    }
    minute_words = {0: "", 15: "fifteen", 30: "thirty", 45: "forty five"}
    hour = hour_words.get(parsed.hour % 12 or 12, str(parsed.hour % 12 or 12))
    minute = minute_words.get(parsed.minute, f"{parsed.minute:02d}")
    ampm = parsed.strftime("%p")
    return f"{hour} {minute} {ampm}".replace("  ", " ").strip()


def _spoken_slots(slots: Any) -> str:
    if not isinstance(slots, list):
        return ""
    spoken = [_spoken_time(slot) for slot in slots if _spoken_time(slot)]
    if not spoken:
        return ""
    if len(spoken) == 1:
        return spoken[0]
    return ", ".join(spoken[:-1]) + f", or {spoken[-1]}"


def _spoken_tool_response(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> str | None:
    if name == "check_availability":
        raw_slots = result.get("open_slots")
        if not isinstance(raw_slots, list) or not raw_slots:
            return "I do not see open slots for that date. Would another day work?"
        preferred_slot = "2:00 PM" if "2:00 PM" in raw_slots else raw_slots[0]
        return (
            f"{_spoken_time(preferred_slot)} on {_human_date(result.get('date'))} is open; "
            "should I book it?"
        )

    if name == "book_appointment":
        confirmation_id = result.get("confirmation_id")
        appointment_date = _human_date(result.get("date"))
        appointment_time = _spoken_time(result.get("time"))
        if confirmation_id:
            return (
                f"Booked for {appointment_date} at {appointment_time}; "
                f"confirmation {confirmation_id}."
            )
        return "I could not complete that booking. The office can help finish it."

    if name == "reschedule_appointment":
        if result.get("status") == "rescheduled":
            return (
                f"You're rescheduled for {_human_date(result.get('date'))} "
                f"at {_spoken_time(result.get('time'))}."
            )
        return "I could not find that confirmation id. The office can help look it up."

    if name == "check_insurance":
        provider = arguments.get("provider") or result.get("provider") or "that plan"
        if result.get("accepted"):
            return f"Yes, Bright Smile Dental accepts {provider}."
        return f"The office will confirm {provider} coverage for you."

    return None


async def _push_spoken_tool_response(params: Any, text: str) -> None:
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection

    await params.llm.push_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await params.llm.push_frame(LLMTextFrame(text), FrameDirection.DOWNSTREAM)
    await params.llm.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)


def register_pipecat_functions(llm: Any) -> None:
    """Register all Bright Smile Dental tool handlers on a Pipecat LLM service."""

    def make_handler(name: str):
        async def handler(params: Any) -> None:
            result = TOOL_IMPLS[name](dict(params.arguments or {}))
            spoken_response = _spoken_tool_response(
                name,
                dict(params.arguments or {}),
                result,
            )
            if spoken_response:
                from pipecat.frames.frames import FunctionCallResultProperties

                async def on_context_updated() -> None:
                    await _push_spoken_tool_response(params, spoken_response)

                await params.result_callback(
                    result,
                    properties=FunctionCallResultProperties(
                        run_llm=False,
                        on_context_updated=on_context_updated,
                    ),
                )
                return

            await params.result_callback(result)

        return handler

    for name in (
        "check_availability",
        "book_appointment",
        "reschedule_appointment",
        "check_insurance",
    ):
        llm.register_function(name, make_handler(name))

    llm.register_function("end_call", _end_call)
