# Improvement Techniques for bot0.py, bot1.py, and bot2.py

This document analyzes the improvement techniques represented by `bot0.py`,
`bot1.py`, and `bot2.py`, then turns those lessons into a systematic plan for
building a better Bright Smile Dental voice agent.

The short version:

- `bot0.py` is the baseline live voice bot.
- `bot1.py` adds coarse end-to-end voice latency measurement.
- `bot2.py` adds deeper latency instrumentation, explicit audio contracts,
  configurable VAD and LLM settings, fast-path booking behavior, and more
  testable construction helpers.
- The most valuable next work is to keep bot2 in a disciplined operating loop:
  measure, inspect stage timings, keep batch concurrency controlled, validate
  behavior in voice mode, and only then keep or revert each change.

## Current Bot Progression

### bot0.py: Baseline Runtime

`bot0.py` establishes the production shape:

```text
transport.input()
  -> NVidiaWebSocketSTTService
  -> LLM user aggregator
  -> VLLMOpenAILLMService
  -> GradiumTTSService
  -> transport.output()
  -> assistant aggregator
```

Core characteristics:

- Supports local browser WebRTC and Twilio websocket transports.
- Uses `NVidiaWebSocketSTTService` for streaming ASR.
- Uses `VLLMOpenAILLMService` for the Nemotron OpenAI-compatible chat endpoint.
- Uses Gradium for TTS.
- Builds the system prompt through `build_system_instruction()`.
- Registers tools through `register_pipecat_functions(llm)`.
- Uses `FilterIncompleteUserTurnStrategies()` in the user aggregator.
- Has no bot-owned latency event log.

The baseline is useful because it is simple and readable, but it gives little
diagnostic leverage. When a call fails, the operator has to infer whether the
problem was VAD, ASR finalization, turn aggregation, LLM delay, tool execution,
TTS, output audio delivery, or the eval harness.

### bot1.py: Measurement Layer

`bot1.py` keeps the same runtime behavior and adds:

```python
latency_logger = LatencyLogger(path=os.getenv("VOICE_LATENCY_LOG_PATH", "latency.jsonl"))
```

It inserts that logger after `transport.output()` and before the assistant
aggregator:

```text
... -> tts -> transport.output() -> LatencyLogger -> assistant_aggregator
```

This is the first major improvement technique: add low-friction runtime
telemetry without changing the behavior being measured.

What it captures:

- `ttfa_ms`: time from `UserStoppedSpeakingFrame` to first outbound audio.
- `ttla_ms`: time from `UserStoppedSpeakingFrame` to last outbound audio.
- `audio_frames` and `audio_bytes`.
- Per-run p50 and p95 summaries.

Why this matters:

- It turns perceived slowness into a number.
- It lets batch evals compare bot versions.
- It distinguishes "no response completed" from "response completed slowly."
- It creates a cheap regression signal for future changes.

Known limitation:

- It measures from the final user-turn stop event, not necessarily from the
  exact acoustic end of speech. If the turn analyzer waits too long, `ttfa_ms`
  can look better than the caller experience. If VAD stops too early, the value
  can look worse or the turn can fragment.

### bot2.py: Tunable Runtime, Stage Instrumentation, and Fast Path

`bot2.py` introduces several higher-leverage changes:

- Constants for WebRTC, Twilio, and STT sample rates.
- `_env_float()` and `_env_int()` helpers for safe environment overrides.
- Separate Twilio and WebRTC VAD parameter builders.
- Explicit `SileroVADAnalyzer(sample_rate=audio_in_sample_rate, params=...)`.
- `build_stt_service()` with an explicit STT sample rate.
- `build_twilio_serializer()` with explicit Twilio input params.
- A timeout around Twilio caller-info lookup.
- LLM temperature and max-token controls.
- Stage-level latency logging through `StageLatencyLogger`.
- Appointment fast-path handling for common booking turns before the LLM.
- Testable helper functions covered by `test_bot2.py`.

The bot2 pipeline is:

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

This is the second major improvement technique: split the voice path into
observable stages so each latency and reliability problem has a narrower search
space.

Earlier local artifacts showed a real bot2 failure mode: latency looked good in
some runs, but conversations could fail to progress past the greeting or split
one caller request into partial turns. The latest bot2 pass addresses the main
root causes directly: it keeps WebRTC VAD conservative, removes LLM-gated
incomplete-turn filtering, makes Twilio/STT sample rates explicit, adds
stage-level artifacts per scenario, handles common booking requests before
streamed LLM tool-call behavior, and makes batch evaluation lower-concurrency by
default.

## Improvement Principles

### 1. Measure Before Optimizing

Voice agents fail in ways that are hard to debug from transcripts alone. A
single final transcript can hide several different root causes:

- VAD started late and cut off the first word.
- VAD stopped too early and split one sentence into many turns.
- ASR finalized quickly but with missing prefixes.
- The turn aggregator waited for incomplete-turn checks.
- The LLM emitted a tool call but no spoken follow-up.
- TTS produced audio, but the client disconnected before it was heard.
- The eval harness timed out while the bot was still working.

The correct technique is to add metrics at the smallest stable boundaries:

- Acoustic speech start.
- Acoustic speech stop.
- Final ASR transcript.
- User turn stop.
- First LLM text or tool call.
- First output audio.
- Last output audio.
- End, cancel, interruption, and superseded statuses.

`bot1.py` starts this discipline with coarse voice latency. `bot2.py` continues
it with stage timings.

Recommended practice:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py bot2.py \
  --scenario happy_booking_cleaning_next_tuesday \
  --max-workers 1
```

Run targeted scenarios first. Full-batch concurrency can hide resource
contention, endpoint rate limits, and cross-run noise.

### 2. Separate Behavioral Quality from Voice Transport Quality

A dental front-desk bot has at least two independent quality axes:

- Task quality: Did it collect details, use tools correctly, obey safety rules,
  and provide the right confirmation?
- Voice quality: Did it hear the caller, segment turns correctly, respond
  quickly, and avoid talking over the caller?

Use text-mode evals for task quality:

```bash
uv run python eval_runner.py --mode text
```

Use voice-mode or batch evals for transport quality:

```bash
uv run python batch_eval_runner.py --bots bot0.py bot1.py bot2.py
```

This prevents a common mistake: changing prompts to compensate for broken VAD or
ASR. If text mode passes and voice mode fails, fix the voice pipeline first.

### 3. Preserve a Known-Good Baseline

`bot0.py` and `bot1.py` are useful even if they are not final. Keep them as
comparison points until a new bot beats them on both:

- Pass rate across representative scenarios.
- Latency distribution, especially p95.

Do not replace the baseline based only on p50 latency. A voice bot with fast
failed turns is worse than a slower bot that finishes the task.

## Technique 1: Coarse Latency Logging

### What to Do

Add a lightweight Pipecat frame processor that observes completed voice turns and
writes JSONL events. `LatencyLogger` is already the right shape.

It should sit after `transport.output()` because that is the closest simple
proxy for audio that is actually being emitted to the caller.

### What to Record

Each completed turn should record:

- `turn_id`
- `status`
- `ttfa_ms`
- `ttla_ms`
- `audio_duration_ms`
- `audio_frames`
- `audio_bytes`
- wall-clock timestamp

Non-complete statuses should also be recorded:

- `interrupted`
- `cancelled`
- `ended`
- `superseded`

These statuses matter because a run with no completed turns is not the same as a
run with fast completed turns.

### How to Use It

Read p50 and p95 from the latest batch metrics:

```bash
head -n 20 batch_metrics.csv
```

Inspect per-scenario turn events:

```bash
cat batch_artifacts/<batch_id>/<bot_id>/<scenario_id>/latency.jsonl
```

Good signs:

- Most scenarios have at least one completed turn after the caller speaks.
- `ttfa_ms` is low enough for a phone conversation.
- `ttla_ms` is not dominated by rambling responses.
- Interrupted turns have a plausible cause, such as caller barge-in.

Bad signs:

- Many events have zero audio frames.
- Most turns are `interrupted` or `cancelled`.
- Only greetings complete.
- Tail latency is several times larger than p50.

## Technique 2: Stage-Level Latency Logging

### What to Do

Use stage probes around the most important pipeline boundaries:

- After STT, before user aggregation.
- After LLM, before TTS.
- After transport output, before assistant aggregation.

`bot2.py` uses this pattern with three `StageLatencyLogger` instances sharing
one tracker.

### Why It Is Better Than One Latency Number

One end-to-end number cannot tell whether the delay came from:

- VAD stop detection.
- ASR finalization.
- Turn aggregation.
- LLM first token or tool call.
- TTS startup.
- Audio egress.

Stage logging can isolate each component:

- `vad_stopped_to_final_transcript`: ASR finalization delay.
- `vad_stopped_to_user_turn_stopped`: turn aggregation delay.
- `vad_stopped_to_llm_first_text_or_tool`: model/tool startup delay.
- `llm_first_text_or_tool_to_first_output_audio`: TTS and audio-output startup.
- `vad_stopped_to_first_output_audio`: true caller-perceived TTFA proxy.

### How to Interpret It

If `vad_stopped_to_final_transcript` is high:

- Optimize STT hard-reset finalization.
- Check websocket latency to ASR.
- Tune preroll and reset behavior.

If `final_transcript_to_user_turn_stopped` is high:

- Investigate turn analyzer policy.
- Reduce incomplete-turn checks.
- Tune silence thresholds.

If `llm_first_text_or_tool_to_first_output_audio` is high:

- Investigate TTS startup.
- Check response length and sentence chunking.
- Consider earlier sentence flushing if supported by the TTS path.

If many stage events are `superseded`:

- VAD is splitting or restarting turns.
- The caller audio source may contain pauses that are too short for current VAD.
- The eval caller may be speaking while the bot is still producing audio.

## Technique 3: VAD Parameterization

### What to Do

Expose VAD parameters through environment variables, but keep separate defaults
for WebRTC and Twilio:

- `VOICE_WEBRTC_VAD_CONFIDENCE`
- `VOICE_WEBRTC_VAD_START_SECS`
- `VOICE_WEBRTC_VAD_STOP_SECS`
- `VOICE_WEBRTC_VAD_MIN_VOLUME`
- `VOICE_VAD_CONFIDENCE`
- `VOICE_VAD_START_SECS`
- `VOICE_VAD_STOP_SECS`
- `VOICE_VAD_MIN_VOLUME`

This is better than one shared VAD profile because WebRTC browser/eval audio and
Twilio phone audio have different sample rates, codecs, noise patterns, and
pause behavior.

### Why It Matters

VAD tuning controls the most important voice tradeoff:

- Short stop time reduces latency but can split a single utterance.
- Long stop time reduces fragmentation but makes the bot feel slow.
- High confidence and min volume reject noise but can miss quiet speech.
- Low confidence and min volume catch quiet speech but can react to noise or TTS
  leakage.

### Recommended Starting Points

For WebRTC voice evals:

```text
VOICE_WEBRTC_VAD_CONFIDENCE=0.55
VOICE_WEBRTC_VAD_START_SECS=0.12
VOICE_WEBRTC_VAD_STOP_SECS=0.55
VOICE_WEBRTC_VAD_MIN_VOLUME=0.35
```

For Twilio:

```text
VOICE_VAD_CONFIDENCE=0.55
VOICE_VAD_START_SECS=0.12
VOICE_VAD_STOP_SECS=0.25
VOICE_VAD_MIN_VOLUME=0.35
```

The local artifacts showed a run where VAD was set to a more aggressive profile
at runtime:

```text
confidence=0.7 start_secs=0.2 stop_secs=0.2 min_volume=0.6
```

That kind of profile can reduce latency, but it is also likely to miss quiet
speech and split utterances. If the transcript contains fragments like "Hi",
"please", or missing first syllables, prefer lower confidence/min-volume and a
longer WebRTC stop time.

### Validation Criteria

After changing VAD:

- Stage logs should show fewer `superseded` turns.
- Transcripts should preserve the beginning of caller utterances.
- Booking scenarios should not split one request into many partial user turns.
- p95 `vad_stopped_to_first_output_audio` should improve or stay acceptable.
- Pass rate should improve or remain stable.

## Technique 4: Explicit Audio Sample-Rate Contracts

### What to Do

Name all sample rates and make conversion points explicit:

```python
WEBRTC_AUDIO_IN_SAMPLE_RATE = 16000
WEBRTC_AUDIO_OUT_SAMPLE_RATE = 24000
TWILIO_AUDIO_IN_SAMPLE_RATE = 16000
TWILIO_AUDIO_OUT_SAMPLE_RATE = 8000
STT_SAMPLE_RATE = 16000
```

For Twilio, use serializer params that state both sides of the contract:

```python
TwilioFrameSerializer.InputParams(
    twilio_sample_rate=TWILIO_AUDIO_OUT_SAMPLE_RATE,
    sample_rate=TWILIO_AUDIO_IN_SAMPLE_RATE,
)
```

### Why It Matters

The STT service expects 16 kHz PCM. Twilio media streams arrive as 8 kHz u-law.
If the code leaves this implicit, the system can appear to work while degrading
VAD and ASR quality.

Sample-rate mismatches can cause:

- Poor recognition.
- VAD thresholds behaving differently than expected.
- Incorrect audio duration estimates.
- Latency metrics that are hard to compare across transports.

### Validation Criteria

Automated tests should assert:

- Twilio transport overrides input to 16 kHz and output to 8 kHz.
- Twilio serializer params expose 16 kHz Pipecat sample rate and 8 kHz Twilio
  sample rate.
- STT initializes with a 16 kHz contract.
- The run logs print active input, output, and STT sample rates.

`test_bot2.py` already covers the core contract. Keep that style for future
audio changes.

## Technique 5: Turn-Completion Strategy Control

### Current Difference

`bot0.py` and `bot1.py` use:

```python
LLMUserAggregatorParams(
    vad_analyzer=SileroVADAnalyzer(),
    user_turn_strategies=FilterIncompleteUserTurnStrategies(),
)
```

`bot2.py` intentionally removes `FilterIncompleteUserTurnStrategies()` and uses
the default turn strategy:

```python
LLMUserAggregatorParams(vad_analyzer=vad_analyzer)
```

### Tradeoff

Incomplete-turn filtering can help avoid responding to half-finished caller
phrases, but it can also:

- Add LLM work to classify turn completion.
- Insert marker tokens such as completion markers into assistant text if the
  model does not obey the marker protocol.
- Increase prompt size.
- Produce extra assistant turns like "go ahead" prompts.
- Delay responses.

Removing incomplete-turn filtering can reduce latency, but it puts more burden
on VAD and ASR finalization. If VAD stops after every short pause, the bot may
respond to fragments or have its LLM responses interrupted by subsequent caller
audio.

### Recommended Technique

Do not treat incomplete-turn filtering as simply on or off. Validate it by
scenario class:

- For short yes/no insurance questions, default turn strategy may be enough.
- For booking flows with names, dates, times, and reasons, incomplete-turn
  filtering may prevent premature responses.
- For voice evals synthesized by `say`, a slightly longer WebRTC VAD stop time
  may be better than using LLM-based incomplete-turn filtering.
- For real Twilio calls, use a separate profile because caller pauses and noise
  differ from synthetic eval audio.

### Specific Next Experiment

Keep the current bot2 default without incomplete-turn filtering. If it is tested
again, do it as a controlled variant with:

- The same explicit audio contracts.
- The same stage logging and batch artifacts.
- The same LLM token controls.
- WebRTC VAD stop at `0.55` or `0.65`.
- No aggressive confidence/min-volume overrides.
- The booking fast path either enabled in both runs or disabled in both runs.

Keep the winner only if it improves both:

- Completed task pass rate.
- p95 true TTFA.

## Technique 6: STT Finalization and Preroll

### Current STT Behavior

`NVidiaWebSocketSTTService` gates audio sends on VAD:

- It buffers preroll audio before VAD start.
- On `VADUserStartedSpeakingFrame`, it sends preroll and live audio.
- On `VADUserStoppedSpeakingFrame`, it sends a hard reset with
  `finalize=True`.
- It emits finalized `TranscriptionFrame` objects only for hard-reset finals.
- It strips cumulative interim prefixes when configured.

This is a strong architecture because it avoids constantly streaming silence and
uses VAD stop as the signal to finalize the utterance.

### Improvement Techniques

Keep the following properties:

- Preroll should preserve speech onsets.
- Reset should be serialized behind any in-flight audio send.
- Final transcripts should be marked `finalized=True`.
- Empty hard-reset finals should clear the waiting state.
- The service should not store full transcripts in latency logs.

Tune carefully:

- `preroll_seconds`: more preroll protects first syllables but sends more audio.
- VAD start time: shorter values protect speech starts but increase false starts.
- VAD stop time: longer values reduce fragmentation but add latency.
- Interim prefix stripping: enable only against cumulative-interim servers.

### Failure Signs

The STT/VAD stack is probably too aggressive when transcripts show:

- Missing first syllables, such as "ule a routine cleaning" instead of
  "schedule a routine cleaning."
- Isolated fragments, such as "Hi", then the main request, then "please."
- Many `superseded` or `interrupted` stage events with no output audio.
- User turn contexts that contain several partial user messages rather than one
  complete request.

## Technique 7: LLM Latency and Output Control

### Current Controls

`bot2.py` adds:

```python
temperature=_env_float("NEMOTRON_LLM_TEMPERATURE", 0.2)
max_tokens=_env_int("NEMOTRON_LLM_MAX_TOKENS", 240)
```

This is a good technique because voice responses should be concise and
predictable. The system prompt already says replies should be one or two short
sentences. A token ceiling makes that operational.

### Recommendations

Use low temperature:

```text
NEMOTRON_LLM_TEMPERATURE=0.2
```

Use a voice-sized max token limit:

```text
NEMOTRON_LLM_MAX_TOKENS=240
```

The current default is 240 because lower caps improved brevity but could cut off
or weaken streamed tool-call turns in live booking evals. Keep responses concise
through prompt and fast-path behavior first; use the token cap as a guardrail,
not as the only brevity mechanism.

Keep thinking disabled for live voice unless the server is proven to separate
reasoning from user-visible content:

```text
NEMOTRON_ENABLE_THINKING=false
```

Reasoning mode can be acceptable for offline text evals, but for voice it can:

- Increase time to first spoken token.
- Produce chain-of-thought in spoken content if the server lacks a reasoning
  parser.
- Distort TTFB metrics unless the LLM service gates TTFB on content/tool deltas.

### Prompt-Level Techniques

The current prompt already includes the right general constraints:

- Keep replies to one or two short sentences.
- Ask one thing at a time.
- Do not invent insurance coverage.
- Only confirm appointments after `book_appointment` returns a confirmation id.
- Do not give clinical advice.
- Say goodbye and call `end_call` in the same turn when the call is complete.

Additional prompt improvements to consider:

- Tell the model to use a tool as soon as all required fields are available.
- Tell the model not to ask for details already supplied in the same user turn.
- Tell the model to normalize relative dates internally before tool calls.
- Tell the model to ask for exactly one missing booking field at a time.
- Tell the model not to speak marker characters or completion symbols.

Do not add long prompt explanations unless text-mode evals show a task-quality
failure. Long prompts can increase latency and make turn-completion markers less
reliable.

## Technique 8: Tool-Use Reliability

### Current Tool Design

`tools.py` is a good single-source-of-truth design:

- `TOOLS` defines OpenAI-style tool schemas.
- `pipecat_tools_schema()` converts those schemas for Pipecat.
- `TOOL_IMPLS` backs text evals and live Pipecat handlers.
- `register_pipecat_functions()` registers the live handlers.

This avoids prompt/eval/live drift.

### bot2 Appointment Fast Path

`bot2.py` now inserts `AppointmentFastPathProcessor` between the user
aggregator and the LLM. It is intentionally narrow: it only handles common
appointment-booking requests that provide, or can quickly collect, name, date,
time, and reason.

The fast path improves the voice agent in three ways:

- It books simple requests through the same backend tool implementation without
  waiting for Nemotron streamed tool-call behavior.
- It asks one missing-field question at a time, then merges the caller's follow-up
  into pending booking state.
- It recognizes direct name follow-ups such as `Jordan Reed.` after asking for a
  name, which fixes a common pending-booking failure.

It also applies a bounded default for vague afternoon requests by using `2:00 PM`
when the caller supplied the other required booking fields. This should remain a
front-desk optimization, not a general replacement for LLM tool use. Anything
outside the narrow booking pattern still falls through to the normal LLM path.

### Improvement Techniques

Strengthen tool behavior around required fields:

- `check_availability` should require normalized date.
- `book_appointment` should require name, date, time, and reason.
- `reschedule_appointment` should require confirmation id, date, and time.
- Unknown insurance should produce a bounded "office will confirm" answer.

Add validation before storing mock bookings:

- Reject empty dates.
- Reject empty names.
- Normalize dates to ISO format where possible.
- Normalize common spoken times like "two thirty PM" to "2:30 PM."

The current mock backend accepts copied args directly. That is fine for a demo,
but it makes bad tool calls look successful. A stricter mock backend would make
eval failures easier to catch.

### Voice-Specific Tool Issue

Voice-mode evals are black-box and usually cannot see tool calls directly. That
means the spoken transcript must contain enough evidence:

- Confirmation id after booking.
- Correct date/time after relative-date interpretation.
- Safe refusal for clinical questions.
- "The office will confirm" for unknown insurance.

Do not rely on invisible tool calls to pass voice-mode evals.

## Technique 9: Twilio Caller Lookup Timeout

### What bot2.py Improves

`bot2.py` adds a bounded caller-info lookup:

```python
timeout = aiohttp.ClientTimeout(total=timeout_secs)
```

The default is:

```text
TWILIO_CALLER_INFO_TIMEOUT_SECS=0.5
```

This is the right technique for phone calls. Caller ID is useful for logging, but
it must not block the first user interaction.

### Recommended Policy

- Treat caller info as best-effort metadata.
- Never infer patient identity from caller ID alone.
- Never let the Twilio REST lookup block startup beyond a small timeout.
- Log status class, not full error bodies that may contain sensitive details.

## Technique 10: Testable Runtime Construction

### What bot2.py Improves

`bot2.py` extracts runtime construction into helpers:

- `twilio_transport_overrides()`
- `build_twilio_vad_params()`
- `build_webrtc_vad_params()`
- `build_vad_analyzer()`
- `build_user_aggregator_params()`
- `build_stt_service()`
- `build_twilio_serializer()`
- `AppointmentFastPathProcessor`
- Narrow parsers for booking date, time, name, and reason extraction.

This is a major maintainability improvement. It lets tests verify contracts
without starting a live bot or external services.

### What to Test

Continue adding narrow tests for:

- Env parsing fallback on invalid numbers.
- VAD default values.
- VAD override values.
- Twilio serializer params.
- STT sample rate.
- LLM default temperature and token cap.
- Absence or presence of incomplete-turn filtering in specific variants.
- Appointment fast-path behavior for complete requests, missing fields, vague
  afternoon requests, and bare-name follow-ups.
- Stage latency tracker behavior for missing stages, duplicate VAD frames, and
  cancelled turns.

Keep these tests cheap and deterministic. The live ASR, LLM, and TTS services
belong in evals, not unit tests.

## Technique 11: Batch Evaluation Discipline

### Use Controlled Comparisons

When comparing bot versions:

- Use the same scenarios.
- Use the same env vars.
- Use `--max-workers 1` for first diagnosis, or rely on the current default when
  running one scenario per bot.
- Capture per-bot artifacts.
- Compare both pass rate and latency.
- Inspect scenario transcripts, not only aggregate metrics.

Recommended smoke loop:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py bot2.py \
  --scenario happy_booking_cleaning_next_tuesday \
  --max-workers 1
```

Recommended focused booking loop:

```bash
uv run python batch_eval_runner.py \
  --bots bot1.py bot2.py \
  --scenario happy_booking_cleaning_next_tuesday \
  --scenario happy_booking_new_patient_friday \
  --scenario happy_booking_cavity_followup \
  --scenario relative_date_this_friday \
  --scenario relative_date_next_monday \
  --max-workers 1
```

Recommended full regression:

```bash
uv run python batch_eval_runner.py \
  --bots bot1.py bot2.py \
  --max-workers 1
```

Raise concurrency only after the single-worker run is stable.

The batch runner now defaults to `min(total_tasks, bot_count)` workers. For a
single bot, that means scenario runs are serial unless `--max-workers` is
explicitly raised. This matters because the ASR, browser/WebRTC harness, and TTS
session limits can create false failures under concurrency. Keep
`EVAL_TTS_ACTIVE_SESSION_LIMIT` as a separate safety cap for explicit worker
overrides and multi-bot runs.

### What to Read First

For a failed scenario:

1. `results.json`: verdict, transcript, turn count, failure reason.
2. `latency.jsonl`: completed, interrupted, cancelled, and superseded turns.
3. `stage_latency.jsonl`, if available: timing stage that failed or never
   appeared.
4. `bot.log`: ASR finals, LLM contexts, function calls, warnings.
5. `eval.log`: client or judge failures.

### Interpret Aggregate Results Carefully

A bot can show better latency because it stopped responding after the greeting.
That is not an improvement. Require enough completed post-caller turns before
treating latency as meaningful.

Minimum useful acceptance gate:

- No infrastructure failures.
- At least one completed bot response after each scenario's caller request.
- Booking pass rate does not regress.
- p95 TTFA improves or stays within target.
- Stage logs do not show widespread fragmentation.

## Technique 12: Separate Synthetic Eval Tuning from Real Phone Tuning

The voice eval path uses generated caller audio. Real phone calls use Twilio
media. They should not share all thresholds.

For synthetic WebRTC evals:

- Caller audio may have clean speech but unnatural pauses.
- Browser/WebRTC timing may differ from Twilio.
- ASR may transcribe generated voices differently from human callers.

For Twilio:

- Audio arrives as 8 kHz u-law.
- Background noise and phone compression are common.
- Users interrupt more naturally.
- Latency expectations are stricter.

Maintain separate VAD env vars and compare separately. Do not tune Twilio by
only looking at browser evals.

## Specific Findings From the Current Variants

### Latest Verification Snapshot

The latest bot2 fix pass verified the code-level changes with:

```bash
uv run pytest
uv run ruff check .
```

The local result was `49 passed` for the test suite and a clean ruff check. The
targeted voice smoke for `happy_booking_cleaning_next_tuesday` passed with a
booked Tuesday appointment and voice p95 TTFA around 302 ms. A serial
five-scenario booking run then passed four of five scenarios; the
`relative_date_this_friday` miss had no caller audio in the artifacts, so it was
treated as a harness miss and rerun in isolation, where it passed with voice p95
TTFA around 1372 ms. Keep that distinction in future postmortems: a run with no
caller audio is not the same failure class as a bot that heard the request and
responded incorrectly.

### Finding 1: bot1.py Is a Valuable Measurement Baseline

`bot1.py` adds latency logging with minimal behavioral changes. This makes it
the safest comparison point for future work.

Use it as the reference for:

- Whether the eval path is working.
- Whether new instrumentation changes behavior.
- Whether latency changed after a non-latency refactor.

### Finding 2: bot2.py Improves Observability and Contracts

The strongest bot2 techniques are:

- Explicit sample-rate constants.
- Twilio serializer input params.
- Separate VAD profiles.
- Stage-level latency probes.
- LLM token controls.
- Appointment fast-path booking for simple requests.
- Pending booking state that merges direct follow-up answers.
- Twilio caller lookup timeout.
- Serial-by-default batch evaluation for one-bot sweeps.
- Unit-testable helper functions.

These should be kept or ported into the eventual default bot.

### Finding 3: bot2.py's Original Failure Was a Combined Voice-Path Issue

Earlier batch artifacts showed bot2 with low measured latency but poor task
completion. The contributing pattern was not one isolated bug; it combined turn
fragmentation, sample-rate ambiguity on the Twilio path, LLM-gated incomplete
turn filtering, and fragile streamed tool-call behavior:

- Short caller phrases were treated as separate turns.
- LLM generations were interrupted by continued caller audio.
- Some runs reached tool calls only after multiple partial transcripts.
- Several scenario transcripts only captured the greeting and caller request.

The current mitigation is systematic: conservative WebRTC VAD defaults, explicit
Twilio/STT sample-rate contracts, default turn aggregation without incomplete
turn filtering, stage latency artifacts, and a narrow booking fast path that
avoids waiting on LLM tool-call streaming for common appointment requests.

### Finding 4: Environment Overrides Can Dominate Code Defaults

`bot2.py` defines conservative WebRTC defaults, but runtime environment values
can override them. Always log active VAD params and sample rates, and always
record the env profile used for benchmark runs.

Without that, two runs of the same file may not be comparable.

### Finding 5: Some Voice Eval Misses Are Harness Failures

The `relative_date_this_friday` scenario produced a failure in a serial batch
where the bot artifacts showed no caller audio arriving. The same scenario
passed when rerun by itself. Future investigations should check for missing
caller audio before tuning prompts, VAD, or tools.

## Recommended Next Bot Variant

Continue hardening `bot2.py` unless preserving every historical variant is more
important than keeping the runtime surface small. Create a `bot3.py` only if the
team wants a clean experiment without changing the current bot2 behavior. The
recommended implementation profile is:

### Keep From bot2.py

- Explicit sample-rate constants.
- Explicit Twilio serializer params.
- `build_stt_service()` with `sample_rate=16000`.
- `_env_float()` and `_env_int()` with warnings.
- Separate WebRTC and Twilio VAD builders.
- `LatencyLogger`.
- `StageLatencyLogger`.
- `AppointmentFastPathProcessor` for narrow booking requests.
- Twilio caller-info timeout.
- LLM temperature and max-token controls.
- Batch eval default concurrency of one scenario per bot.
- Helper functions with unit tests.

### Change From bot2.py

- Broaden the booking fast path only when the new pattern is deterministic and
  covered by unit tests.
- Keep checking that `.env` does not override WebRTC VAD to aggressive thresholds
  during evals.
- Treat incomplete-turn filtering as a controlled experiment, not a default
  assumption.
- Add a concise prompt rule to avoid asking for already-provided fields outside
  the fast path.
- Add tool-argument validation so empty tool calls do not look successful.
- Add a per-run config snapshot to batch outputs so VAD and LLM settings are
  preserved with metrics.

### Candidate Defaults

```text
NEMOTRON_ENABLE_THINKING=false
NEMOTRON_LLM_TEMPERATURE=0.2
NEMOTRON_LLM_MAX_TOKENS=240

VOICE_WEBRTC_VAD_CONFIDENCE=0.55
VOICE_WEBRTC_VAD_START_SECS=0.12
VOICE_WEBRTC_VAD_STOP_SECS=0.55
VOICE_WEBRTC_VAD_MIN_VOLUME=0.35

VOICE_VAD_CONFIDENCE=0.55
VOICE_VAD_START_SECS=0.12
VOICE_VAD_STOP_SECS=0.25
VOICE_VAD_MIN_VOLUME=0.35
```

## Acceptance Gates

A bot improvement should not be considered successful until it passes these
gates.

### Unit Gate

```bash
uv run pytest test_bot2.py test_metrics.py test_eval_runner.py test_batch_eval_runner.py
```

Required:

- Sample-rate tests pass.
- VAD env override tests pass.
- Latency tracker tests pass.
- Batch result aggregation tests pass.

### Text Eval Gate

```bash
uv run python eval_runner.py --mode text
```

Required:

- Tool behavior is correct.
- Relative dates are interpreted correctly.
- Guardrails pass.
- Booking and rescheduling flows produce expected confirmation behavior.

### Voice Smoke Gate

```bash
uv run python batch_eval_runner.py \
  --bots bot1.py bot2.py \
  --scenario happy_booking_cleaning_next_tuesday \
  --max-workers 1
```

Required:

- No infrastructure failure.
- The bot responds after the caller request.
- `latency.jsonl` has a completed post-caller turn.
- Transcript includes meaningful booking progress.

### Focused Booking Gate

```bash
uv run python batch_eval_runner.py \
  --bots bot1.py bot2.py \
  --scenario happy_booking_cleaning_next_tuesday \
  --scenario happy_booking_new_patient_friday \
  --scenario happy_booking_cavity_followup \
  --scenario relative_date_this_friday \
  --scenario relative_date_next_monday \
  --max-workers 1
```

Required:

- Pass rate is at least as good as bot1 on the same env.
- p95 TTFA is not worse than bot1.
- No widespread `superseded` or `interrupted` zero-audio turns.

### Full Voice Gate

```bash
uv run python batch_eval_runner.py \
  --bots bot1.py bot2.py \
  --max-workers 1
```

Required:

- Booking, rescheduling, insurance, safety, policy, and call-closure scenarios
  stay stable.
- Latency distribution is stable across more than one run.
- Failures are explainable from artifacts.

## Implementation Checklist

Use this checklist when making the next improvement pass.

- Confirm `.env` does not contain stale aggressive VAD overrides.
- Run `git status --short` before editing.
- Preserve `bot1.py` as a reference implementation.
- Add or update helper functions instead of duplicating inline setup code.
- Add tests for every new env var or audio contract.
- Keep latency logs payload-free; do not store caller transcript text in metrics.
- Log active sample rates and VAD params at startup.
- Run text mode after prompt or tool changes.
- Run voice mode after VAD, STT, TTS, transport, or aggregator changes.
- Compare pass rate and p95 latency together.
- Treat "fast but no task progress" as a failure.

## Priority Roadmap

### Priority 1: Preserve Turn Segmentation and Booking Progress

Goal:

- The bot should hear a full caller request as one coherent turn unless the
  caller truly pauses long enough to invite a response, and common booking
  requests should progress without waiting on fragile streamed tool calls.

Actions:

- Reset WebRTC VAD env vars to conservative defaults.
- Re-run five booking scenarios serially or with one worker per bot.
- Inspect stage logs for `superseded` and `interrupted` turns.
- Keep the booking fast path narrow and covered by unit tests.
- Tune `VOICE_WEBRTC_VAD_STOP_SECS` upward before reintroducing LLM incomplete
  turn filtering.

### Priority 2: Keep bot2 Observability

Goal:

- Every voice failure should identify the missing or slow stage.

Actions:

- Keep stage loggers.
- Ensure batch artifacts copy `stage_latency.jsonl`, not only root-level logs.
- Add stage-latency summaries to `batch_metrics.csv`.
- Add a per-run config snapshot for VAD, sample rates, LLM settings, ASR URL
  host, and bot SHA.

### Priority 3: Tighten Tool Validation

Goal:

- Bad tool calls should fail visibly instead of silently creating weak results.

Actions:

- Validate required fields in tool implementations.
- Return structured error statuses for missing values.
- Teach the prompt to repair missing fields by asking the caller one question.
- Add text eval cases for empty date, missing name, and invalid time.

### Priority 4: Reduce LLM and TTS Tail Latency

Goal:

- Maintain task quality while reducing p95 response delay.

Actions:

- Keep `NEMOTRON_LLM_MAX_TOKENS=240`; lower it only after booking evals show
  tool-call and confirmation quality remain stable.
- Keep responses to one spoken sentence when asking for missing details.
- Investigate TTS first-audio latency from
  `llm_first_text_or_tool_to_first_output_audio`.
- Avoid LLM incomplete-turn filtering unless it clearly raises pass rate enough
  to justify latency cost.

### Priority 5: Improve Eval Reliability

Goal:

- Failures should distinguish agent behavior from infrastructure.

Actions:

- Run low-concurrency first.
- Preserve all artifacts per scenario.
- Add stage latency artifacts to each scenario directory.
- Keep separate `judge_error`, `infrastructure_failure`, and agent-failure
  categories.
- Require repeated runs before accepting small latency deltas.

## Final Recommendation

The best path is to keep `bot1.py` as the stable measured baseline and treat
`bot2.py` as the actively hardened voice-runtime branch. Bot2 now contains the
right shape for the next default candidate: explicit audio contracts, conservative
VAD defaults, stage artifacts, bounded startup work, concise LLM settings, and a
narrow deterministic booking fast path.

The next high-confidence improvement pass should keep:

- bot1's stable behavior and latency logger.
- bot2's explicit audio contracts.
- bot2's stage latency probes.
- conservative WebRTC VAD defaults.
- bot2's appointment fast path for common bookings.
- serial-by-default batch evaluation for one-bot sweeps.
- per-run config snapshots.
- stricter tool validation.

That combination gives the project the right operating loop: every change has a
measured effect, every failure has an artifact trail, and performance work cannot
accidentally trade away the front-desk task.
