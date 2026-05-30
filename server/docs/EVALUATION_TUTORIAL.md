# Evaluation Runner

This repo includes an eval runner for Bright Smile Dental. By default it runs
against a locally hosted Pipecat voice bot at `http://localhost:7860`.

The voice path:

- Starts a WebRTC session through the local Pipecat runner.
- Synthesizes caller turns with macOS `say`.
- Streams caller audio into the bot.
- Captures the bot's spoken audio and transcribes it with the NVIDIA ASR websocket.
- Judges the resulting transcript with the configured Nemotron LLM.

The legacy text harness is still available with `--mode text`. It exercises the
same prompt and `tools.py` implementations in-process, but it does not test audio,
STT, TTS, WebRTC transport, or turn-taking.

## Setup

Install dependencies:

```bash
uv sync
```

Start a local bot in another terminal:

```bash
uv run bot.py
```

The runner expects the bot at `http://localhost:7860`. You can point it at any
other Pipecat local runner on that port or override the URL:

```bash
export EVAL_BOT_URL="http://localhost:7860"
```

Configure the LLM used for simulated caller turns and judging:

```bash
export NEMOTRON_LLM_URL="http://192.168.7.228:8000/v1"
export NEMOTRON_LLM_MODEL="nvidia/nemotron-3-super"
export NEMOTRON_LLM_API_KEY="EMPTY"
export NEMOTRON_ENABLE_THINKING="false"
export NEMOTRON_LLM_TIMEOUT="60"
```

Configure ASR for transcribing the bot's spoken responses:

```bash
export NVIDIA_ASR_URL="ws://192.168.7.228:8081"
```

## Run Voice Evals

Smoke test one scenario:

```bash
uv run python eval_runner.py --limit 1
```

Run the full scenario set:

```bash
uv run python eval_runner.py
```

Run one scenario:

```bash
uv run python eval_runner.py --scenario reschedule_valid_id
```

Useful voice options:

```bash
uv run python eval_runner.py \
  --bot-url http://localhost:7860 \
  --response-timeout 45 \
  --caller-rate 185
```

## Run Text Evals

Use the old in-process text harness when you only want fast prompt/tool checks:

```bash
uv run python eval_runner.py --mode text --limit 2
```

## Outputs

`results.json` contains the latest run:

- Run metadata including `eval_mode`, model, scenario count, pass count, and pass rate.
- `bot_url` for voice-mode runs.
- `voice_p95_latency_ms` and `voice_latency` if `latency.jsonl` exists.
- Per-scenario verdicts with id, pass/fail, judge reason, turn count, transcript, and
  captured tool calls when available.

`runs.jsonl` appends one compact trend row per eval run for dashboarding.

Voice-mode evals treat the bot as a black box, so tool calls are usually empty. The
judge uses a lenient answer-present policy: if the needed answer, action,
confirmation, or safe refusal appears anywhere in the spoken transcript, the scenario
should pass even when wording, order, or captured tool logs are imperfect.

## Verification

Recommended checks after changing prompt, tools, scenarios, or eval code:

```bash
uv run pytest
uv run ruff check .
uv run pyright .
uv run python eval_runner.py --limit 1
uv run python eval_runner.py --mode text --limit 2
```
