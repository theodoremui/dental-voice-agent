# Bright Smile Dental Voice Agent

This folder documents the Bright Smile Dental front-desk voice agent and the
evaluation system around it.

The numbered bot files in the server directory, `bot0.py`, `bot1.py`, `bot2.py`,
and `bot3.py`, are not four unrelated demos. They are an engineering sequence.
Each version keeps the same core product goal, then improves one part of the
voice-agent stack: correctness, latency, observability, deterministic behavior,
and front-desk usefulness.

## The Product Goal

Bright Smile Dental needs a front-desk assistant that can answer routine calls
without pretending to be a dentist, a scheduler with unlimited authority, or a
patient-records system. The agent is named Aria. Its job is narrow on purpose:

- Greet callers naturally.
- Book dental appointments after collecting the required details.
- Reschedule existing appointments when the caller has a valid confirmation ID.
- Check whether a caller's insurance provider is in the known accepted list.
- Decline tasks it should not perform, such as cancellation or diagnosis.
- Route urgent symptoms toward emergency care first.
- Keep calls short, spoken, and front-desk-like.

The voice agent is useful only if it behaves like a competent front desk. That
means it must be fast enough to feel conversational, smart enough to gather the
right information, and safe enough to avoid clinical or privacy mistakes.

## The Core Contract

The business rules live in `tools.py`. The most important runtime prompt is built
by `build_system_instruction()`:

```python
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
...
"""
```

The prompt is intentionally practical. It does not tell the model to be generally
helpful in every possible way. It tells the model what a dental front desk can and
cannot do.

The same file defines the mock backend used by local development and evaluation:

```python
TOOL_IMPLS = {
    "check_availability": _check_availability,
    "book_appointment": _book_appointment,
    "reschedule_appointment": _reschedule_appointment,
    "check_insurance": _check_insurance,
    "end_call": _end_call_text_eval,
}
```

This is a key design choice. The model can speak, but the business state changes
through tools. Booking is not just a sentence. Booking is a tool call that returns
a confirmation ID.

## What The Tools Mean

### `check_availability`

The front desk can check open appointment slots for a requested date:

```python
def _check_availability(args: dict[str, Any]) -> dict[str, Any]:
    appointment_date = args.get("date", "")
    return {"date": appointment_date, "open_slots": ["1:00 PM", "2:00 PM", "2:30 PM", "4:00 PM"]}
```

This supports callers who say things like "next Tuesday afternoon" without
forcing the model to invent slots.

### `book_appointment`

The front desk can book only after it has name, date, time, and reason:

```python
{
    "name": "book_appointment",
    "description": (
        "Book a dental appointment. Call only after collecting name, date, time, "
        "and reason."
    ),
    "parameters": {
        "required": ["name", "date", "time", "reason"],
    },
}
```

This is the central workflow. The voice agent should not say "you are booked"
until the tool returns a confirmation.

### `reschedule_appointment`

The agent can reschedule an existing appointment only when the caller provides a
confirmation ID:

```python
def _reschedule_appointment(args: dict[str, Any]) -> dict[str, Any]:
    confirmation_id = args.get("confirmation_id")
    if not isinstance(confirmation_id, str) or confirmation_id not in _BOOKINGS:
        return {"status": "not_found", "confirmation_id": confirmation_id}
    _BOOKINGS[confirmation_id].update(args)
    return {"status": "rescheduled", "confirmation_id": confirmation_id, **args}
```

This keeps the agent from guessing patient identity or claiming it found records
it cannot actually access.

### `check_insurance`

The accepted insurance list is deliberately small:

```python
known = {"delta dental", "metlife", "aetna"}
```

For unknown providers, the right answer is not a hallucinated yes. The right
answer is that the office will confirm coverage.

### `end_call`

Ending the call is also a tool. The prompt tells the agent to say goodbye and
call `end_call` in the same turn. That makes call closure observable in tests.

## The Voice Pipeline

All numbered bots use the same basic shape:

```text
transport.input()
  -> speech-to-text
  -> user turn aggregation
  -> front-desk intelligence
  -> text-to-speech
  -> transport.output()
```

In code, the baseline pipeline in `bot0.py` is:

```python
pipeline = Pipeline(
    [
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ]
)
```

Each step has a different job:

- `transport.input()` receives audio from browser WebRTC or Twilio.
- `NVidiaWebSocketSTTService` turns caller audio into text.
- `LLMContextAggregatorPair` converts partial speech into usable model context.
- `VLLMOpenAILLMService` runs Nemotron through an OpenAI-compatible endpoint.
- `GradiumTTSService` turns Aria's text into speech.
- `transport.output()` sends spoken audio back to the caller.
- `assistant_aggregator` records the assistant side of the conversation.

The later bots keep this mental model, but add instrumentation and deterministic
front-desk behavior around it.

## Why Voice Agents Are Harder Than Text Agents

A text chatbot waits for complete user messages. A voice agent has to decide when
the caller has stopped speaking, often while the caller is hesitating, correcting
themselves, or speaking over background noise.

The Bright Smile Dental agent has to handle all of these:

- "I want next Friday. Actually, the Friday after that."
- "Afternoon is fine. What times do you have?"
- "My name is Jamie Lee. Sorry, Jamie Li."
- "Do you take Cigna? Another dentist does."
- "I have facial swelling and severe pain."

The user experience depends on both intelligence and timing. A correct answer
that arrives too late feels broken. A fast answer that ignores the caller is also
broken.

## Evolution Of The Bots

## `bot0.py`: Baseline Live Voice Agent

`bot0.py` establishes the first working product shape:

- Browser WebRTC support.
- Twilio websocket support.
- NVIDIA streaming speech-to-text.
- Nemotron LLM with tool calling.
- Gradium text-to-speech.
- Bright Smile Dental system prompt and tools.
- Caller ID lookup for Twilio logging only.

The important privacy detail is in the Twilio path:

```python
# Fetch call information from Twilio REST API for logging only.
# Do not infer patient identity from caller ID.
call_info = await get_call_info(call_data["call_id"])
if call_info:
    from_number = call_info.get("from_number")
```

The system prompt reinforces that rule:

```python
caller_context = (
    "Twilio supplied caller ID for logging only. Do not infer patient identity, claim "
    "to recognize the caller, or mention caller records based on phone number alone."
    if from_number
    else "No caller ID context is available. Treat this as a standard front-desk call."
)
```

This version is valuable because it proves the whole stack can run. It also gives
the team a simple baseline to compare against.

The limitation is observability. If a call feels slow or fails halfway through,
`bot0.py` does not tell us where the time went. Was it VAD? STT? LLM? Tool
execution? TTS? The baseline cannot answer that.

## `bot1.py`: The Same Agent, Now Measurable

`bot1.py` adds the first important operational improvement: latency logging.

```python
latency_logger = LatencyLogger(path=os.getenv("VOICE_LATENCY_LOG_PATH", "latency.jsonl"))
```

It inserts the logger after outbound audio:

```python
pipeline = Pipeline(
    [
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        latency_logger,
        assistant_aggregator,
    ]
)
```

This is a disciplined change. It keeps behavior almost identical to `bot0.py`,
but makes the voice experience measurable.

The latency tracker records:

```python
event["ttfa_ms"] = round((turn.first_audio_at - turn.stopped_speaking_at) * 1000)
event["ttla_ms"] = round((turn.last_audio_at - turn.stopped_speaking_at) * 1000)
```

`ttfa_ms` means time to first audio. This is the caller's wait before Aria starts
speaking.

`ttla_ms` means time to last audio. This captures the full spoken response length
and delivery time.

`bot1.py` does not make Aria smarter by itself. It makes improvement possible.
Before `bot1.py`, the team could only say "this feels slow." After `bot1.py`,
the team can compare p50 and p95 latency across bot versions and scenarios.

## `bot2.py`: Faster Runtime, Stage Visibility, And A Narrow Fast Path

`bot2.py` changes from "measure the whole turn" to "understand the stages inside
the turn."

It adds explicit runtime constants:

```python
WEBRTC_AUDIO_IN_SAMPLE_RATE = 16000
WEBRTC_AUDIO_OUT_SAMPLE_RATE = 24000
TWILIO_AUDIO_IN_SAMPLE_RATE = 16000
TWILIO_AUDIO_OUT_SAMPLE_RATE = 8000
STT_SAMPLE_RATE = 16000
```

It also makes VAD configurable:

```python
def build_webrtc_vad_params() -> VADParams:
    return VADParams(
        confidence=_env_float("VOICE_WEBRTC_VAD_CONFIDENCE", DEFAULT_WEBRTC_VAD_CONFIDENCE),
        start_secs=_env_float("VOICE_WEBRTC_VAD_START_SECS", DEFAULT_WEBRTC_VAD_START_SECS),
        stop_secs=_env_float("VOICE_WEBRTC_VAD_STOP_SECS", DEFAULT_WEBRTC_VAD_STOP_SECS),
        min_volume=_env_float("VOICE_WEBRTC_VAD_MIN_VOLUME", DEFAULT_WEBRTC_VAD_MIN_VOLUME),
    )
```

This matters because VAD is the front door of the voice agent. If VAD stops too
soon, it splits one caller sentence into fragments. If it waits too long, the
agent feels slow. The best setting depends on the channel.

`bot2.py` also adds stage-level latency loggers:

```python
stage_transcript_logger = StageLatencyLogger(
    path=os.getenv("VOICE_STAGE_LATENCY_LOG_PATH", "stage_latency.jsonl"),
    observe_upstream=True,
    observe_turn=True,
    observe_transcript=True,
    observe_llm=False,
    observe_audio=False,
    finalize=False,
)
```

The pipeline becomes:

```text
transport.input()
  -> stt
  -> stage_transcript_logger
  -> user_aggregator
  -> appointment_fast_path
  -> llm
  -> stage_turn_logger
  -> tts
  -> transport.output()
  -> latency_logger
  -> stage_audio_logger
  -> assistant_aggregator
```

This lets the team separate:

- When VAD thought the caller started speaking.
- When VAD thought the caller stopped speaking.
- When the final transcript arrived.
- When the user turn was considered complete.
- When the LLM first produced text or a tool call.
- When the first output audio appeared.
- When the last output audio appeared.

That is the difference between debugging by transcript and debugging by system
behavior.

### The Appointment Fast Path

`bot2.py` introduces `AppointmentFastPathProcessor`, a small deterministic
processor for common booking requests:

```python
class AppointmentFastPathProcessor(FrameProcessor):
    """Handles simple appointment turns without waiting on LLM tool-call streaming."""
```

It reads the latest user text, extracts simple structured fields, and can answer
before the LLM is needed:

```python
booking = {
    "name": _extract_name(user_text) or "",
    "date": _extract_relative_or_absolute_date(user_text, self._today) or "",
    "time": _extract_time(user_text) or "",
    "reason": _extract_reason(user_text) or "",
}
```

When it has enough information, it calls the same backend tool:

```python
result = TOOL_IMPLS["book_appointment"](self._pending_booking)
return (
    f"Booked for {_format_fast_date(result['date'])} at {_format_fast_time(result['time'])}; "
    f"confirmation {confirmation_id}."
)
```

This is not a replacement for the LLM. It is a speed path for predictable
front-desk work. The LLM remains available for messy cases.

The tradeoff is real. A fast path can make calls much faster, but only if it is
narrow and well tested. If it tries to handle everything, it becomes a second
agent hidden inside the pipeline. `bot2.py` keeps the fast path focused on
appointments.

## `bot3.py`: Memory-First Front Desk

`bot3.py` generalizes the `bot2.py` idea. Instead of a narrow appointment fast
path, it introduces a broader memory-first front-desk processor:

```python
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
```

This is a major product improvement. Real front-desk conversations are stateful.
Callers correct themselves, answer only part of a question, repeat information,
or ask a follow-up after the agent offered options. `CallMemory` gives the agent
a local working memory for one call.

The processor sits before the LLM:

```python
front_desk_memory = MemoryFirstFrontDeskProcessor()

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
```

The design is "memory first, LLM fallback." If the caller's turn matches a
well-understood front-desk workflow, `bot3.py` answers from state and tools. If
not, the frame continues to the LLM.

The processor decides its path in `_response_for_context()`:

```python
if self._looks_like_insurance(lowered):
    return self._insurance_response(lowered)

if self._looks_like_reschedule(lowered) or self._memory.intent == "reschedule":
    self._memory.intent = "reschedule"
    return self._reschedule_response()

if self._looks_like_booking(lowered) or self._memory.has_booking_context():
    self._memory.intent = "booking"
    return self._booking_response(lowered)

return None
```

Returning `None` is important. It means "I do not know how to handle this
deterministically, so let the LLM handle it." This keeps deterministic code from
overreaching.

### Better Booking Behavior

`bot3.py` remembers partial booking details:

```python
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
    ...
```

This is more useful than repeatedly asking for the same details. It also handles
corrections:

```python
is_correction = _looks_like_correction(normalized)

appointment_date = self._date_from_text(normalized)
if appointment_date and (not self._memory.appointment_date or is_correction or self._memory.intent):
    self._memory.appointment_date = appointment_date
```

This directly supports scenarios like "next Friday, actually the Friday after
that."

### Better Time Handling

When a caller asks for an afternoon appointment without a specific time, the
agent should not invent a time silently. It should either check availability or
offer slots.

`bot3.py` tracks that:

```python
if _mentions_vague_afternoon(normalized) and not appointment_time:
    self._memory.vague_time_requested = True
if any(phrase in lowered for phrase in ("what times", "which times", "options", "open slots")):
    self._memory.wants_time_options = True
```

Then it can offer open slots:

```python
def _offer_slots(self) -> str:
    self._memory.offered_slots = self._available_slots()
    if not self._memory.offered_slots:
        return "I do not see open slots that day. Would another date work?"
    return (
        f"I have {_format_slot_list(self._memory.offered_slots)} on "
        f"{_format_fast_date(self._memory.appointment_date)}. Which works?"
    )
```

This turns the agent from a prompt-following assistant into a more operational
front-desk flow.

### Safer Rescheduling

`bot3.py` treats rescheduling as a stateful workflow:

```python
if not self._memory.reschedule_confirmation_id:
    self._memory.last_question = "confirmation_id"
    return "What is your confirmation ID?"
if not self._memory.appointment_date:
    self._memory.last_question = "date"
    return "What date should I move it to?"
if not self._memory.appointment_time:
    self._memory.last_question = "time"
    return "What time should I move it to?"
```

It then calls `reschedule_appointment`. If the ID is invalid, it does not pretend
the appointment moved:

```python
if result.get("status") == "rescheduled":
    return (
        f"You're rescheduled for {_format_fast_date(result['date'])} "
        f"at {_format_fast_time(result['time'])}."
    )
return "I could not find that confirmation ID. The office can help look it up."
```

This is exactly the kind of boundary a dental front desk needs.

### Safer Policy Handling

`bot3.py` also handles common policy and safety issues before the LLM:

```python
if any(word in lowered for word in ("severe pain", "facial swelling", "trauma", "bleeding heavily")):
    return "Please seek emergency care first. I can also help schedule an urgent dental visit."
if any(phrase in lowered for phrase in ("ibuprofen", "root canal", "diagnose", "what dose")):
    return "I cannot give dental or medication advice, but I can help book a dentist visit."
if "cancel" in lowered:
    return "I cannot cancel appointments here. The office can help, or I can help reschedule."
if "phone number" in lowered or "pull up my chart" in lowered or "know who i am" in lowered:
    return "I cannot identify you or pull up records from caller ID alone."
```

This makes the agent more useful because it becomes reliably limited. A safe
front-desk agent should be good at saying no in the right way.

## Faster, Better, More Intelligent, More Useful

The bot sequence improves along four dimensions.

### Faster

`bot1.py` makes latency visible. `bot2.py` and `bot3.py` reduce unnecessary LLM
round trips for common front-desk workflows. A deterministic response from a
processor can be spoken without waiting for the LLM to reason through tool use.

The speed improvement is not only about milliseconds. It is about reducing
uncertainty in the call path. If the agent can book a simple appointment from
structured memory, there are fewer moving parts that can fail.

### Better

The later bots are better because they preserve business rules more explicitly.
They understand that booking requires name, date, time, and reason. They
understand that insurance answers must stay inside the known list. They
understand that rescheduling requires a confirmation ID.

The system becomes less dependent on the LLM remembering every rule on every
turn.

### More Intelligent

The intelligence in this codebase is not only in the LLM. It is in the division
of labor:

- The LLM handles language flexibility.
- Tools handle business state.
- VAD and aggregators handle conversation timing.
- Fast-path processors handle deterministic workflows.
- Per-call memory tracks what has already been collected.
- Evaluations define what "good" means.

That architecture is more intelligent than asking one model prompt to do
everything.

### More Useful

The later bots are more useful because they behave closer to a real front-desk
assistant:

- They remember that the caller already gave their name.
- They can offer appointment slots.
- They can handle a caller correction.
- They can reschedule safely.
- They can answer insurance questions without overclaiming.
- They can refuse medical advice without ending the conversation.
- They can tell emergency callers to seek care first.

Usefulness here means "helps the caller complete safe front-desk work."

## Evaluation Harness

The canonical scenario set is in `scenarios.py`. It currently covers 25 cases
across:

- Booking.
- Rescheduling.
- Insurance.
- Medical safety.
- Policy guardrails.
- Call closure.

Each scenario has a caller persona and success criteria:

```python
{
    "id": "happy_booking_cleaning_next_tuesday",
    "category": "booking",
    "severity": "high",
    "persona": (
        "You are Maria Lopez. Book a routine cleaning for next Tuesday afternoon. "
        "Give your name, date, time, and reason when asked."
    ),
    "criteria": (
        "Agent checked availability, collected name/date/time/reason, booked the cleaning, "
        "and gave a confirmation id."
    ),
}
```

This is the quality bar. A bot version is not better because it has more code or
faster first audio. It is better if it passes more of these scenarios while
keeping latency acceptable.

Run a small comparison:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py bot2.py bot3.py \
  --limit 3 \
  --max-workers 1
```

Run a targeted safety scenario:

```bash
uv run python batch_eval_runner.py \
  --bots bot2.py bot3.py \
  --scenario medical_advice_ibuprofen \
  --max-workers 1
```

Review the generated output:

```bash
jq '.bots | to_entries[] | {bot: .key, pass_rate: .value.pass_rate, latency: .value.voice_latency}' batch_results.json
```

## How To Read Results

Do not optimize only for p50 latency. The right comparison is:

- Pass rate.
- Failing scenario IDs.
- Judge reasons.
- Infrastructure failure count.
- `ttfa_p95_ms`.
- `ttla_p95_ms`.
- Stage timings, when available.

A fast bot that only greets the caller is not successful. A slower bot that books
correctly may be more useful. The goal is to move both curves: higher pass rate
and lower caller-perceived delay.

## Design Lessons From The Evolution

### Keep The Prompt Small But Clear

The system prompt should define policy and tone, not encode every deterministic
branch. Deterministic branches belong in tools or processors when they are stable
enough to test.

### Make State Changes Tool-Backed

Appointments and reschedules should be tool-backed because callers need reliable
confirmation. A spoken claim is not enough.

### Use Memory For One Call, Not Identity

`bot3.py` has per-call memory. That is different from patient identity. It can
remember "the caller said 2:30 PM" during one call. It must not infer "this is
Maria Lopez" from caller ID.

### Add Fast Paths Carefully

Fast paths are powerful when they handle predictable work. They are dangerous
when they silently become a second untested conversation engine. The correct
pattern is:

- Keep the rule narrow.
- Reuse the same backend tools.
- Return `None` when the processor is unsure.
- Let the LLM handle open-ended language.
- Cover the behavior with tests and scenarios.

### Instrument Before Rewriting

The move from `bot0.py` to `bot1.py` is important because it avoids guessing.
Without latency events, teams often rewrite prompts when the real issue is VAD,
STT finalization, or TTS startup.

### Separate Voice Quality From Task Quality

Text-mode evals are useful for reasoning and tool behavior. Voice-mode evals are
necessary for turn-taking, ASR, VAD, TTS, and latency.

Both matter.

## Recommended Working Loop

1. Pick one scenario family, such as booking or insurance.
2. Run `bot1.py`, `bot2.py`, and `bot3.py` with `--max-workers 1`.
3. Compare pass rate and p95 latency.
4. Inspect failed transcripts and stage latency files.
5. Improve the narrowest failing layer.
6. Add or update tests when the fix is deterministic.
7. Run the full 25-scenario batch only after the narrow scenario passes.

Example:

```bash
uv run python batch_eval_runner.py \
  --bots bot1.py bot2.py bot3.py \
  --scenario ambiguous_time_afternoon \
  --max-workers 1
```

Then inspect:

```bash
jq '.bots.bot3.scenarios[0] | {passed, reason, transcript}' batch_results.json
```

## Where The Project Is Heading

The long-term direction is not "more LLM everywhere." The better direction is a
hybrid voice agent:

- LLM for natural language flexibility.
- Tools for committed business actions.
- Per-call memory for context.
- Deterministic processors for stable high-volume workflows.
- Observability for latency and stage timing.
- Evaluation harnesses for scenario quality.
- Cekura for external scenario management and call-log evaluation when needed.

That is what the bot sequence is teaching. `bot0.py` proves the live call stack.
`bot1.py` measures it. `bot2.py` instruments and accelerates it. `bot3.py`
starts turning the agent into an actual front-desk workflow system.

The best Bright Smile Dental agent will not be the version with the cleverest
single prompt. It will be the version where every layer does the job it is best
suited to do.
