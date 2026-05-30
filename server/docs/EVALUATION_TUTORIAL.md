# Voice Evaluation Tutorial

`eval_runner.py` tests the actual Pipecat voice bot already running at
`http://localhost:7860`. It does not import the bot's prompt/tools, does not start
`bot.py`, and does not use `BOT_EVAL_SERVER`.

Start the bot in another terminal first:

```bash
uv run bot.py
```

Then run evals from `server/`:

```bash
uv run python eval_runner.py --scenario office_hours
```

The runner connects through the Pipecat WebRTC runner:

1. `GET /status` verifies that the running service supports WebRTC.
2. `POST /start` creates a real WebRTC session.
3. The simulated caller's text turns are synthesized to speech and sent over the audio track.
4. The bot's returned audio is captured from WebRTC and transcribed for judging.
5. A judge LLM scores the voice transcript against the scenario criteria.

## Configuration

The bot under test is selected with:

```bash
export EVAL_BOT_URL=http://localhost:7860
```

The caller simulator and judge still use an OpenAI-compatible chat endpoint:

```bash
export EVAL_BASE_URL=http://192.168.7.228:8000/v1
export EVAL_API_KEY=EMPTY
export EVAL_MODEL=nvidia/nemotron-3-super
```

The evaluator needs caller TTS and transcription for the bot audio. By default it uses
Gradium for caller speech and the NVIDIA ASR websocket for transcription:

```bash
export GRADIUM_API_KEY=...
export EVAL_TRANSCRIBE_ASR_URL=ws://192.168.7.228:8081
```

Optional caller voice override:

```bash
export EVAL_CALLER_TTS_VOICE=Eu9iL_CYe8N-Gkx_
```

On macOS, you can use the local `say` command for caller speech:

```bash
uv run python eval_runner.py --caller-tts-provider say --caller-tts-voice Samantha
```

## Scenarios

List scenarios:

```bash
uv run python eval_runner.py --list
```

Run one:

```bash
uv run python eval_runner.py --scenario happy_booking_next_tuesday
```

Run a subset:

```bash
uv run python eval_runner.py --scenario office_hours,emergency_swelling
```

Run all:

```bash
uv run python eval_runner.py
```

## Batch Comparison

Use `batch_eval_runner.py` to compare multiple voice-agent implementations in one run:

```bash
uv run python batch_eval_runner.py --bots bot0.py bot1.py bot2.py
```

The batch runner starts each bot with a separate Pipecat WebRTC port, then runs
`eval_runner.py` against each bot URL in parallel. By default it uses every bot as a
parallel job; cap concurrency if the shared STT, TTS, or LLM services are saturated:

```bash
uv run python batch_eval_runner.py --bots bot0.py bot1.py bot2.py --jobs 2
```

Forward normal `eval_runner.py` filters after the batch options:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --scenario office_hours,emergency_swelling \
  --max-turns 4
```

Outputs:

- `batch_results.json` contains the full aggregate result and embeds each per-bot
  `results.json` with `evaluated_agent.bot`, `bot_path`, `bot_port`, and `bot_url`.
- `batch_runs.jsonl` appends one trend row per bot with `pass_rate`,
  `p95_agent_reply_ms`, `p95_latency_ms`, `live_p95_ttfa_ms`, and `live_p95_ttla_ms`.
- `batch_eval_artifacts/<batch_id>/<bot>/` stores per-bot result, latency, and process
  logs for debugging.

## Boundary

These are live voice-path evals, so they exercise WebRTC, STT, the bot's LLM/tool behavior,
TTS, and returned audio transcription. Because the runner only observes the external voice
interface, internal tool calls are not generally available unless the bot exposes them over
the transport. The judge uses the spoken transcript as the primary evidence.
