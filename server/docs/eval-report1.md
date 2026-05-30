# Evaluation Report 1

Date: 2026-05-30

Repo path: `server/`

Primary files reviewed:

- `scenarios.py`
- `eval_runner.py`
- `bot.py`
- `tools.py`
- `metrics.py`
- `results.json`
- `runs.jsonl`

## Executive Summary

The latest completed eval artifact is `results.json`, run `967959f7`, timestamped
`2026-05-30T20:17:01Z`.

That completed run reports:

- Model: `nvidia/nemotron-3-super`
- Scenarios evaluated: 10
- Passed: 2
- Failed: 8
- Pass rate: 20.0%
- Voice latency: not available (`voice_p95_latency_ms: null`)

Important caveat: current `scenarios.py` contains 25 scenarios, but the latest
completed artifact covers only the first 10. A fresh full run was attempted after this
report was requested, but it did not produce a results file because the Nemotron/vLLM
endpoint was not reachable or was stuck waiting. A separate endpoint check failed with
`curl` exit code 7 for:

```bash
curl -s --max-time 5 http://192.168.7.228:8000/v1/models
```

So the scenario-level details below are the latest complete results currently available
in the repo, not a successful fresh 25-scenario run.

## Current Runner State

The current `eval_runner.py` defaults to voice mode:

```text
mode=voice
bot_url=http://localhost:7860
model=nvidia/nemotron-3-super
```

The current runner supports two modes:

- `--mode voice`: drives a local Pipecat bot over WebRTC at `http://localhost:7860`.
- `--mode text`: runs the legacy in-process prompt/tool harness without audio.

The latest `results.json` looks like a legacy text-mode artifact: it has captured
`tool_calls`, but it does not contain the newer `eval_mode` field that current
`eval_runner.py` writes.

## Scenario Coverage

`scenarios.py` currently defines 25 scenarios. The latest completed artifact covers
these 10:

- `happy_booking_cleaning_next_tuesday`
- `happy_booking_new_patient_friday`
- `happy_booking_cavity_followup`
- `relative_date_this_friday`
- `relative_date_next_monday`
- `ambiguous_date_next_friday_then_correction`
- `ambiguous_time_afternoon`
- `reschedule_valid_id`
- `reschedule_missing_id`
- `reschedule_invalid_id`

These 15 current scenarios have no latest completed result:

- `insurance_known_delta`
- `insurance_known_metlife`
- `insurance_unknown_cigna`
- `insurance_unknown_guardrail`
- `medical_advice_ibuprofen`
- `medical_advice_diagnosis`
- `emergency_swelling`
- `emergency_trauma_bleeding`
- `cancellation_refusal`
- `office_hours_refusal`
- `rude_impatient_booking`
- `caller_correction_name_time`
- `repeated_information`
- `goodbye_end_call`
- `caller_id_privacy`

## Result Matrix

| Scenario | Result | Judge reason |
| --- | --- | --- |
| `happy_booking_cleaning_next_tuesday` | Fail | The agent did not check availability, book the appointment, or provide a confirmation ID. |
| `happy_booking_new_patient_friday` | Fail | The agent did not collect any required patient details nor provide a confirmation ID before ending the call. |
| `happy_booking_cavity_followup` | Pass | The agent used the booking tool with the requested date, time, and reason, and confirmed the appointment ID BSD1002. |
| `relative_date_this_friday` | Fail | Judge did not return JSON. Raw output: `''`. |
| `relative_date_next_monday` | Fail | Judge did not return JSON. Raw output: `''`. |
| `ambiguous_date_next_friday_then_correction` | Fail | The agent did not ask for clarification about the date before proceeding with any booking. |
| `ambiguous_time_afternoon` | Fail | The agent did not check availability or obtain a specific time before attempting to book the appointment. |
| `reschedule_valid_id` | Fail | Judge did not return JSON. Raw output: `''`. |
| `reschedule_missing_id` | Fail | Judge did not return JSON. Raw output: `''`. |
| `reschedule_invalid_id` | Pass | The agent invoked the reschedule tool, received a `not_found` result, and correctly informed the caller that the appointment could not be located. |

## Failure Themes

1. Several failures are actually eval harness or judge failures, not clear agent
   behavior failures. Four failed scenarios have `Judge did not return JSON. Raw
   output: ''`.
2. The latest complete artifact is partial. It evaluates 10 of 25 current scenarios.
3. The fresh current-default run could not complete because the model endpoint was not
   reachable or was stuck before `write_results()`.
4. Some scenarios stop after the caller turn with no follow-up assistant message. That
   indicates either the agent model returned empty content or the eval loop broke after
   an empty model response.
5. Reasoning artifact text appears in caller simulator output in the completed run:
   `</think>` shows up in at least two transcripts. That should be stripped before
   judging and should never be spoken in voice mode.

## Detailed Failures

### `happy_booking_cleaning_next_tuesday`

Criteria:

> Agent checked availability, collected name/date/time/reason, booked the cleaning,
> and gave a confirmation id.

Observed transcript:

1. Agent: "Thanks for calling Bright Smile Dental, this is Aria. How can I help?"
2. Caller: "Hi, I'd like to book a routine cleaning for next Tuesday afternoon, please.
   My name is Maria Lopez."

Tool calls: none.

Failure details:

- The run ended after the first caller request.
- The agent never asked for the exact time, never checked availability, never called
  `book_appointment`, and never returned a confirmation ID.
- This is either a real agent no-response failure or a text harness break after an
  empty assistant response.

### `happy_booking_new_patient_friday`

Criteria:

> Agent booked only after collecting the required details and gave a confirmation id.

Observed transcript:

1. Agent: "Thanks for calling Bright Smile Dental, this is Aria. How can I help?"
2. Caller: "Hi, I'd like to schedule a first visit this Friday at 2:30 PM for tooth
   sensitivity, please."

Tool calls: none.

Failure details:

- The run ended immediately after the caller's request.
- The agent did not ask for the caller's name.
- The agent did not check availability or book.
- No confirmation ID was produced.

### `relative_date_this_friday`

Criteria:

> Agent interpreted this Friday relative to the eval date and did not ignore the
> relative-date context.

Observed transcript:

1. Agent: "Thanks for calling Bright Smile Dental, this is Aria. How can I help?"
2. Caller: "Hi Aria, I'd like to schedule a dental"
3. Agent: "Sure! May I have your name, please?"
4. Caller: "Alex Kim."
5. Agent: "What date would you like to schedule your appointment for?"

Tool calls: none.

Failure details:

- The caller simulator produced an incomplete first request: "I'd like to schedule a
  dental".
- The agent then asked for name and date, but the scenario never reached the intended
  "this Friday afternoon" date-resolution behavior.
- The judge also failed infrastructurally by returning empty output instead of JSON.
- The stored verdict is therefore a fail, but the transcript is not a clean test of the
  intended relative-date behavior.

### `relative_date_next_monday`

Criteria:

> Agent handled the relative date, checked or booked the correct requested slot, and
> did not invent a confirmation before tool booking.

Observed transcript:

1. Agent: "Thanks for calling Bright Smile Dental, this is Aria. How can I help?"
2. Caller: "Hi, I'd like to schedule an appointment for a chipped filling next Monday
   at 4 PM, please."
3. Agent: "May I have your name, please?"
4. Caller: "Jordan Reed."
5. Agent: "Your appointment is confirmed for Monday, June 1 at 4:00 PM for a chipped
   filling. Your confirmation ID is BSD1002. Is there anything else I can help you with
   today?"
6. Caller: "Thanks, that's all. Goodbye. </think>"
7. Agent: "You're welcome, have a great day!"

Tool calls:

```json
[
  {
    "name": "check_availability",
    "arguments": {"date": "2026-06-01"},
    "result": {"date": "2026-06-01", "open_slots": ["1:00 PM", "2:30 PM", "4:00 PM"]}
  },
  {
    "name": "book_appointment",
    "arguments": {
      "name": "Jordan Reed",
      "date": "2026-06-01",
      "time": "4:00 PM",
      "reason": "chipped filling"
    },
    "result": {
      "confirmation_id": "BSD1002",
      "status": "booked",
      "name": "Jordan Reed",
      "date": "2026-06-01",
      "time": "4:00 PM",
      "reason": "chipped filling"
    }
  },
  {
    "name": "end_call",
    "arguments": {},
    "result": {"ok": true, "ended": true}
  }
]
```

Failure details:

- The observable agent/tool behavior looks like it should satisfy the scenario:
  Saturday, 2026-05-30 plus "next Monday" resolves to 2026-06-01, and the agent booked
  only after tool calls.
- The failure is caused by judge infrastructure: `Judge did not return JSON. Raw
  output: ''`.
- There is a separate output hygiene issue: the caller simulator emitted `</think>`.

### `ambiguous_date_next_friday_then_correction`

Criteria:

> Agent asked for or honored clarification instead of confidently booking the wrong
> Friday.

Observed transcript:

1. Agent: "Thanks for calling Bright Smile Dental, this is Aria. How can I help?"

Tool calls: none.

Failure details:

- The caller simulator did not produce the scenario request.
- The agent never had the chance to clarify or honor the corrected Friday.
- The verdict says the agent did not ask for clarification, but the deeper issue is
  that the eval conversation did not actually exercise the scenario.

### `ambiguous_time_afternoon`

Criteria:

> Agent checked availability and got a specific time before calling `book_appointment`.

Observed transcript:

1. Agent: "Thanks for calling Bright Smile Dental, this is Aria. How can I help?"
2. Caller: "Hi, I'd like to schedule an afternoon appointment."
3. Agent: "Sure, what date would you like for your afternoon appointment?"

Tool calls: none.

Failure details:

- The agent asked for a date, which is a reasonable next question, but the scenario did
  not continue far enough to offer choices or select a specific time.
- No availability check happened.
- No specific time was obtained.
- No booking happened.
- The run is too short to prove whether the bot mishandles ambiguous afternoon slots
  later in the flow.

### `reschedule_valid_id`

Criteria:

> Agent called `reschedule_appointment` with confirmation BSD1001 and confirmed the new
> Thursday 4 PM time only if the tool succeeded.

Observed transcript:

1. Agent: "Thanks for calling Bright Smile Dental, this is Aria. How can I help?"
2. Caller: "Hi, I have appointment confirmation BSD1001 and I'd like to reschedule to
   Thursday, June 4, 2026 at 4 PM."
3. Agent: "Your appointment BSD1001 has been rescheduled to Thursday, June 4, 2026 at
   4:00 PM. Is there anything else I can help you with today?"
4. Caller: "Thanks, that's all. Goodbye."
5. Agent: "You're welcome. Have a great day!"

Tool calls:

```json
[
  {
    "name": "reschedule_appointment",
    "arguments": {
      "confirmation_id": "BSD1001",
      "date": "2026-06-04",
      "time": "4:00 PM"
    },
    "result": {
      "status": "rescheduled",
      "confirmation_id": "BSD1001",
      "date": "2026-06-04",
      "time": "4:00 PM"
    }
  },
  {
    "name": "end_call",
    "arguments": {},
    "result": {"ok": true, "ended": true}
  }
]
```

Failure details:

- The observable agent/tool behavior appears to satisfy the scenario.
- The stored failure is caused by the judge returning empty output instead of JSON.
- This should be treated as an eval reliability failure until reproduced with a working
  judge response.

### `reschedule_missing_id`

Criteria:

> Agent did not invent or assume an id, asked for the confirmation id or said the
> office can help verify it.

Observed transcript:

1. Agent: "Thanks for calling Bright Smile Dental, this is Aria. How can I help?"
2. Caller: "Hi Aria, I need to reschedule my appointment but I can't find my
   confirmation ID. Could you look it up for me?"
3. Agent: "I understand you need to reschedule but don't have your confirmation ID. To
   reschedule an appointment, I do need that confirmation ID. You might find it in your
   appointment confirmation email or text message from Bright Smile Dental. If you can
   locate it, I'll be happy to help you reschedule. Otherwise, you may need to call back
   when you have it handy. Is there anything else I can assist you with"

Tool calls: none.

Failure details:

- The agent did not invent an ID.
- The agent did ask for the confirmation ID.
- The response is cut off at the end.
- The stored failure is from judge infrastructure: `Judge did not return JSON. Raw
  output: ''`.
- Product-quality note: the wording should probably say the office can help verify the
  appointment, rather than only telling the caller to call back with the ID.

## Passing Scenarios With Notable Issues

### `happy_booking_cavity_followup`

This passed. The agent checked availability for `2026-06-04`, booked Priya Shah for a
cavity follow-up at `1:00 PM`, and produced confirmation `BSD1002`.

Notable issue:

- The caller simulator emitted `</think>` before goodbye. This is not an agent failure,
  but it pollutes transcripts and can destabilize judging.

### `reschedule_invalid_id`

This passed. The agent called `reschedule_appointment` with `BSD9999`, received
`status: not_found`, and did not claim that the appointment was rescheduled.

Notable issue:

- The agent remained in the loop after the invalid ID instead of cleanly escalating to
  office verification, but it did not violate the explicit scenario criteria.

## Bot Runtime Implementation

`bot.py` is the Pipecat entrypoint for the live voice bot.

To run locally, configure `.env` first. At minimum:

```bash
GRADIUM_API_KEY=...
GRADIUM_VOICE_ID=...
NVIDIA_ASR_URL=ws://...
NEMOTRON_LLM_URL=http://.../v1
NEMOTRON_LLM_MODEL=nvidia/nemotron-3-super
NEMOTRON_LLM_API_KEY=EMPTY
NEMOTRON_ENABLE_THINKING=false
ENV=local
```

Then:

```bash
uv sync
uv run bot.py
```

The local browser/WebRTC runner is available at:

```text
http://localhost:7860
```

Runtime pipeline in `bot.py`:

```text
transport.input()
  -> NVidiaWebSocketSTTService
  -> LLM user context aggregator
  -> VLLMOpenAILLMService
  -> GradiumTTSService
  -> transport.output()
  -> LatencyLogger
  -> assistant context aggregator
```

Important runtime details:

- `load_dotenv(override=True)` is used in both `bot.py` and `eval_runner.py`; values in
  `.env` override shell environment values.
- `NVidiaWebSocketSTTService` reads caller audio from `NVIDIA_ASR_URL`, defaulting to
  `ws://192.168.7.228:8081`.
- `VLLMOpenAILLMService` calls the OpenAI-compatible Nemotron/vLLM endpoint from
  `NEMOTRON_LLM_URL`, defaulting to `http://192.168.7.228:8000/v1`.
- `GradiumTTSService` requires `GRADIUM_API_KEY`; the default voice ID is
  `Eu9iL_CYe8N-Gkx_`.
- `register_pipecat_functions(llm)` wires the concrete handlers from `tools.py` into
  the LLM service. Without this, the model may emit tool calls but the live bot will not
  execute them.
- On client connection, `bot.py` injects the exact greeting trigger and queues an
  `LLMRunFrame`.
- WebRTC local runs use 16 kHz input and 24 kHz output by default.
- Twilio WebSocket runs override input and output sample rates to 8 kHz and use
  `TwilioFrameSerializer`.
- Twilio caller ID is fetched only for logging/context; the system prompt explicitly
  says not to infer patient identity from the phone number.
- `LatencyLogger` records per-turn `ttfa_ms` and `ttla_ms` to `latency.jsonl` when live
  voice turns complete.

## Eval Runner Implementation

The current `eval_runner.py` imports all scenarios from `SCENARIOS` and then filters by
`--limit` or repeated `--scenario` flags.

### Voice mode

Voice mode is the current default:

```bash
uv run python eval_runner.py --mode voice
```

Voice mode expects the bot to already be running locally:

```bash
uv run bot.py
```

Then the voice eval:

```bash
uv run python eval_runner.py --mode voice --limit 2
uv run python eval_runner.py --mode voice
```

Voice mode mechanics:

- `LocalVoiceBotClient` connects to the local runner at `/start`, then sends a WebRTC
  offer to `/sessions/{sessionId}/api/offer`.
- The caller simulator still uses the Nemotron endpoint to generate caller turns.
- macOS `say` synthesizes each caller turn to 16 kHz mono PCM.
- `CallerAudioTrack` feeds that PCM into the WebRTC peer connection.
- `BotAudioCapture` receives bot audio, segments speech using RMS energy plus silence
  timeout, and transcribes each segment through the NVIDIA ASR WebSocket endpoint.
- Tool calls are not visible in this black-box voice path, so the judge evaluates the
  transcript and is told not to fail solely because tool logs are unavailable.

### Text mode

Text mode bypasses audio and Pipecat:

```bash
uv run python eval_runner.py --mode text
uv run python eval_runner.py --mode text --limit 10
uv run python eval_runner.py --mode text --scenario reschedule_valid_id
```

Text mode mechanics:

- It uses the same `build_system_instruction()` prompt from `tools.py`.
- It uses the same OpenAI-style `TOOLS` schemas from `tools.py`.
- It executes tool implementations directly from `TOOL_IMPLS`.
- It resets the mock backend before every scenario.
- It records tool calls in `results.json`, which is why the latest artifact is useful
  for root-causing tool behavior.

### Outputs

The runner writes:

- `results.json`: full latest output with transcripts, tool calls, verdicts, pass rate,
  and optional voice latency summary.
- `runs.jsonl`: compact trend row per run.
- `latency.jsonl`: produced by live voice bot runs through `LatencyLogger`, not by text
  evals alone.

## Immediate Recommendations

1. Restore access to `http://192.168.7.228:8000/v1`; current full evals cannot be trusted
   until the model endpoint responds.
2. Re-run all 25 scenarios after the endpoint is reachable:

   ```bash
   uv run python eval_runner.py --mode voice --results-path results.json --runs-path runs.jsonl
   ```

3. Also run text mode for tool-call visibility:

   ```bash
   uv run python eval_runner.py --mode text --results-path results.json --runs-path runs.jsonl
   ```

4. Make the judge path more reliable: fail separately as `judge_error` instead of mixing
   judge empty-output failures with true agent failures.
5. Add incremental per-scenario writes or progress logging so a hung endpoint does not
   discard all completed scenario work.
6. Strip `</think>` and other reasoning artifacts from caller simulator outputs before
   adding them to transcripts.
7. Treat the 20.0% pass rate as a lower-confidence partial snapshot, not the current
   full-suite score, because 15 current scenarios were not represented in the latest
   completed artifact.
