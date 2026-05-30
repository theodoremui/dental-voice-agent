# AGENT.md

## Repo Summary

This repo is a Bright Smile Dental voice-agent project built on Pipecat. The active dental agent is a Python server in `server/` that can run over local browser WebRTC or Pipecat/Twilio websocket media streams.

The current production/deploy target is `server/bot2.py`: `server/Dockerfile` copies `bot2.py` into the image as `bot.py`. `server/bot0.py` and `server/bot1.py` are earlier variants. `server/bot3.py` and `server/test_bot3.py` are currently untracked working-tree files and appear to be an experimental memory-first successor to `bot2.py`.

The top-level `README.md` is still mostly the hackathon starter README and mentions flower-shop and `bot.py` / `bot-nemotron.py` flows that do not match the current dental code exactly. Prefer this file plus `server/docs/*.md` for current repo orientation.

## Main Runtime Pieces

- `server/bot2.py`: active deployed dental bot. Pipeline: transport input -> NVIDIA streaming STT -> stage latency logger -> Pipecat user context aggregator -> appointment fast path -> Nemotron/vLLM LLM -> stage latency logger -> Gradium TTS -> transport output -> latency loggers -> assistant context aggregator.
- `server/tools.py`: source of truth for Aria's system prompt, OpenAI-style tool schemas, Pipecat tool schema conversion, tool registration, and the in-memory dental backend.
- `server/nvidia_stt.py`: custom Pipecat websocket STT service for NVIDIA ASR. It gates audio on VAD, buffers preroll audio, sends hard resets on VAD stop, and emits finalized transcripts so user turns can close quickly.
- `server/nemotron_llm.py`: OpenAI-compatible vLLM service wrapper that measures TTFB at the first visible content/tool delta instead of reasoning-only chunks.
- `server/metrics.py`: voice latency and stage latency processors. `latency.jsonl` tracks TTFA/TTLA from caller stop to outbound audio; `stage_latency.jsonl` tracks VAD, transcript, LLM, and audio milestones.
- `server/eval_runner.py`: single-run eval harness. Voice mode drives a local Pipecat bot over WebRTC; text mode runs an in-process prompt/tool harness.
- `server/batch_eval_runner.py`: launches bot files on temporary local ports and runs voice scenarios in parallel for bot-to-bot comparisons.
- `server/dashboard.py`: Streamlit dashboard utilities for reading `results.json`, `runs.jsonl`, batch results, scenario summaries, and latency trends.
- `server/scenarios.py`: 25 current evaluation scenarios across booking, rescheduling, insurance, medical safety, policy guardrails, and call closure.

## Supported User-Facing Features

Aria is the front-desk assistant for Bright Smile Dental. The prompt and tools currently support:

- Appointment availability lookup for a date.
- Appointment booking after collecting patient name, date, time, and reason.
- Appointment rescheduling with an existing Bright Smile Dental confirmation ID.
- Insurance checks for the known accepted list: Delta Dental, MetLife, and Aetna.
- Unknown insurance handling by saying the office will confirm coverage.
- Medical-safety refusal for diagnosis, treatment, and medication questions.
- Emergency handling for severe pain, facial swelling, trauma, or bleeding by directing the caller to emergency care first, then offering an urgent visit.
- Caller-ID privacy: Twilio caller ID is for logging only and must not be used to claim identity or chart access.
- Clean goodbye handling via the `end_call` tool in the LLM path.

The mock backend is intentionally small:

- Availability always returns afternoon slots: `1:00 PM`, `2:00 PM`, `2:30 PM`, `4:00 PM`.
- Bookings are in memory only and reset between eval scenarios.
- A demo booking `BSD1001` exists for reschedule tests.
- There is no real patient lookup, calendar, chart, insurance API, cancellation backend, office-hours source, or persistence.

## Bot Variants

- `bot0.py`: baseline dental Nemotron voice bot. Uses NVIDIA STT, Nemotron LLM, Gradium TTS, tool registration, WebRTC/Twilio transports, and `FilterIncompleteUserTurnStrategies`.
- `bot1.py`: `bot0.py` plus `LatencyLogger` after `transport.output()`.
- `bot2.py`: current deploy target. Adds explicit sample-rate contracts, Twilio serializer input params, env-tunable VAD, bounded Twilio caller-info lookup, lower default LLM temperature/max tokens, `StageLatencyLogger`, and `AppointmentFastPathProcessor`.
- `bot3.py`: untracked experimental variant. Adds `MemoryFirstFrontDeskProcessor` with per-call `CallMemory` for booking, rescheduling, insurance, and policy responses before LLM fallback. It handles fragmented details, corrections, vague afternoon slot offers, known policy responses, and some rescheduling flows without waiting for LLM tool-call streaming.
- `bot-gpt.py`: original Field & Flower flower-shop starter using Gradium STT/TTS and OpenAI Responses. It is not part of the Bright Smile Dental behavior surface.

## Current Working Behavior

What appears solid in code and unit tests:

- Tool schemas and tool implementations are centralized in `tools.py`.
- Spoken tool responses avoid a post-tool LLM hop for availability, booking, rescheduling, and insurance in the Pipecat LLM tool path.
- `bot2.py` has tests for Twilio sample-rate handling, VAD defaults/env overrides, removal of incomplete-turn filtering, LLM token defaults, and the simple appointment fast path.
- `bot3.py` has tests for memory across turns, corrections, vague afternoon slot offering, policy responses, and insurance responses.
- Eval output writing captures pass rates, category summaries, transcripts, tool calls, judge policy metadata, voice-agent metadata, and latency summaries.
- Batch evals reserve local ports, launch short-lived bot processes, wait on `/status`, write per-task artifacts, and aggregate bot-level results.

What has worked in stored artifacts:

- `bot1.py` reached a 0.8 pass rate on one five-scenario voice batch, but with high p95 TTFA/TTLA in that run.
- `bot2.py` showed much lower p95 TTFA in the latest stored five-scenario batch, but the same run failed all five booking scenarios because the voice eval transcripts often captured only the greeting and caller turn.
- Earlier `results.json` shows voice-mode booking sometimes produced spoken confirmations, but there were infrastructure and judging failures mixed into the results.

Treat stored eval artifacts as evidence, not ground truth. Several runs are partial, use different judge policies, or show voice capture/turn-taking failures rather than pure agent-behavior failures.

## Known Limitations And Risks

- The active deploy image uses `bot2.py`, not `bot3.py`. `bot3.py` is untracked and not copied by the Dockerfile.
- `bot2.py`'s `AppointmentFastPathProcessor` is intentionally narrow. It only recognizes a small set of dates, times, names, reasons, and booking-like phrases. It can book directly but does not cover the full scenario set.
- `bot3.py`'s memory-first processor is broader but still regex-based. It handles common eval phrases, not arbitrary natural language.
- Voice-mode evals are black-box. Captured tool calls are usually empty, so the judge grades spoken outcomes rather than internal tool use.
- The judge is an LLM. The runner has JSON repair and a lenient answer-present policy, but judge failures still appear in old artifacts.
- The voice eval path depends on macOS `say` for caller speech synthesis.
- Runtime depends on external services: Gradium TTS, NVIDIA ASR websocket, and Nemotron/vLLM. If those endpoints are unreachable or slow, evals and local calls fail or hang.
- `NEMOTRON_ENABLE_THINKING` defaults off for voice. Keep it off unless the vLLM server is confirmed to separate reasoning from spoken content; otherwise reasoning may appear in `content` and be spoken.
- Twilio caller lookup is best-effort and must remain privacy-safe. The prompt explicitly forbids inferring identity from caller ID.
- Office hours and cancellations are unsupported by design.
- No production persistence exists. Bookings/reschedules use process-local dictionaries.
- `README.md` and some docs mention stale filenames (`bot.py`, `bot-nemotron.py`) or starter content. Check actual files before following those commands.

## Environment

Run commands from `server/` unless noted.

Required for local bot/evals:

```bash
uv sync
export GRADIUM_API_KEY="..."
export GRADIUM_VOICE_ID="..."          # optional; defaults in code if unset
export NVIDIA_ASR_URL="ws://..."
export NEMOTRON_LLM_URL="http://.../v1"
export NEMOTRON_LLM_MODEL="nvidia/nemotron-3-super"
export NEMOTRON_LLM_API_KEY="EMPTY"
export NEMOTRON_ENABLE_THINKING="false"
export ENV="local"
```

Useful latency and tuning env vars:

```bash
export VOICE_LATENCY_LOG_PATH="latency.jsonl"
export VOICE_STAGE_LATENCY_LOG_PATH="stage_latency.jsonl"
export NEMOTRON_LLM_TEMPERATURE="0.2"
export NEMOTRON_LLM_MAX_TOKENS="240"
export TWILIO_CALLER_INFO_TIMEOUT_SECS="0.5"
export VOICE_VAD_CONFIDENCE="0.55"
export VOICE_VAD_START_SECS="0.12"
export VOICE_VAD_STOP_SECS="0.25"
export VOICE_VAD_MIN_VOLUME="0.35"
export VOICE_WEBRTC_VAD_STOP_SECS="0.55"
```

Twilio path also uses:

```bash
export TWILIO_ACCOUNT_SID="..."
export TWILIO_AUTH_TOKEN="..."
```

## Common Commands

Install and test:

```bash
cd server
uv sync
uv run pytest
uv run ruff check .
uv run pyright .
```

Run the active local bot:

```bash
cd server
uv run bot2.py
```

Open `http://localhost:7860` and connect over WebRTC.

Run the experimental memory-first bot:

```bash
cd server
uv run bot3.py
```

Run a single voice eval against a bot already listening on port 7860:

```bash
cd server
uv run python eval_runner.py --mode voice --limit 1
```

Run fast text harness checks:

```bash
cd server
uv run python eval_runner.py --mode text --limit 2
```

Compare bot variants by launching them automatically:

```bash
cd server
uv run python batch_eval_runner.py --bots bot1.py bot2.py --limit 5 --max-workers 2
```

Run the dashboard:

```bash
cd server
uv run streamlit run dashboard.py
```

Deploy config:

- `server/pcc-deploy.toml` uses `agent_name = "dental-bot"` and `secret_set = "dental-bot-secrets"`.
- `server/Dockerfile` currently deploys `bot2.py` by copying it to `bot.py`.

## Guidance For Future Agents

- Preserve the tool source of truth in `tools.py`; update prompt rules, OpenAI-style schemas, Pipecat schema conversion, and mock implementations together.
- If changing booking behavior, add focused unit tests around `AppointmentFastPathProcessor` or `MemoryFirstFrontDeskProcessor` before relying on voice evals.
- Keep voice latency instrumentation after `transport.output()` if you want TTFA/TTLA to represent accepted outbound audio frames.
- Be careful interpreting voice eval failures. Inspect per-scenario `transcript`, `bot.log`, `eval.log`, `latency.jsonl`, and `stage_latency.jsonl` before deciding whether the problem is the agent, VAD/STT/TTS, the eval harness, or the judge.
- Do not turn on Nemotron thinking for voice unless the server-side reasoning parser is verified.
- Do not rely on caller ID for identity or records. This is both a prompt rule and a product limitation.
- If `bot3.py` is intended to be the new active bot, update `Dockerfile`, docs, and batch comparisons explicitly.
