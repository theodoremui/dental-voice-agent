# Batch Evaluation Runner

This guide explains how to compare multiple Bright Smile Dental voice-agent
implementations with the batch eval runner.

The canonical command is:

```bash
uv run python batch_eval_runner.py --bots bot0.py bot1.py
```

For compatibility, this equivalent form is also supported:

```bash
uv run python eval_runner.py --bots bot0.py bot1.py
```

Both commands run voice-mode evals. The batch runner starts each bot file as a
short-lived local Pipecat process on its own localhost port, runs one scenario
against that process, stores the artifacts, stops the bot, and moves on. Tasks
run in parallel up to `--max-workers`.

## What Batch Mode Tests

Batch mode treats each bot as a black-box voice agent:

- Launches each bot with `uv run <bot>.py --host localhost --port <port>`.
- Waits for `GET /status` to report that the bot is ready.
- Runs `eval_runner.py --mode voice` against that bot URL.
- Synthesizes caller speech with macOS `say`.
- Captures and transcribes bot audio through the configured NVIDIA ASR service.
- Judges the resulting transcript with the configured Nemotron LLM.
- Aggregates pass rate, judge errors, infrastructure failures, and voice latency.

Batch mode is the right path when you want to compare implementations such as
`bot0.py`, `bot1.py`, and `bot-gpt.py` across the same scenario set.

## Prerequisites

Install dependencies from the `server` directory:

```bash
uv sync
```

Make sure the local environment can run the bots:

```bash
cp .env.example .env
```

Configure the LLM used for simulated callers and judging:

```bash
export NEMOTRON_LLM_URL="http://192.168.7.228:8000/v1"
export NEMOTRON_LLM_MODEL="nvidia/nemotron-3-super"
export NEMOTRON_LLM_API_KEY="EMPTY"
export NEMOTRON_ENABLE_THINKING="false"
export NEMOTRON_LLM_TIMEOUT="60"
```

Configure ASR for transcribing spoken bot responses:

```bash
export NVIDIA_ASR_URL="ws://192.168.7.228:8081"
```

You can put the same variables in `.env`; both `batch_eval_runner.py` and
`eval_runner.py --bots ...` load `.env` before reading defaults. Shell exports
still work, but `.env` is the simplest way to keep the batch runner and bot
processes on the same ASR, LLM, and TTS endpoints.

Configure any runtime variables required by the bots themselves, such as:

```bash
export GRADIUM_API_KEY="..."
export GRADIUM_VOICE_ID="..."
export ENV="local"
```

The voice eval path uses macOS `say` for caller audio synthesis, so run voice
batch evals on macOS unless you replace that synthesis path.

## Quick Start

Run one scenario against two bots:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --limit 1 \
  --max-workers 1
```

Run the same smoke test through the compatibility entrypoint:

```bash
uv run python eval_runner.py \
  --bots bot0.py bot1.py \
  --limit 1 \
  --max-workers 1
```

Run the full scenario set:

```bash
uv run python batch_eval_runner.py --bots bot0.py bot1.py
```

Run a specific scenario:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --scenario reschedule_valid_id
```

Run multiple named scenarios:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --scenario insurance_unknown_cigna \
  --scenario emergency_swelling \
  --scenario caller_correction_name_time
```

## Concurrency and Ports

By default, `--max-workers` is:

```text
min(total_tasks, min(6, len(bots) * 2))
```

For example, with 2 bots and 25 scenarios, the default is 4 workers. With 4 bots
and 25 scenarios, the default is 6 workers.

Use a lower worker count when local CPU, TTS, ASR, or LLM capacity is limited:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --max-workers 2
```

If Gradium returns `Concurrency limit exceeded: 3 active sessions`, lower
`--max-workers`. A two-bot comparison is usually safest at `--max-workers 2`
unless the TTS account has a higher concurrency limit.

Use a higher worker count only if the local machine and remote services can handle
the load:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py bot-gpt.py \
  --max-workers 6
```

The runner starts searching for free ports at `--base-port`, which defaults to
`7860`. If a port is already in use, it skips to the next available port.

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --base-port 9000
```

You do not need to start the bots yourself. Batch mode starts and stops them for
each bot/scenario task.

## Timeouts

Startup timeout controls how long the runner waits for each bot process to expose
`GET /status`:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --startup-timeout 90
```

Eval timeout controls the maximum time allowed for one scenario run:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --eval-timeout 900
```

Response timeout controls how long `eval_runner.py` waits for each spoken bot
response:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --response-timeout 60
```

Silence timeout controls how much bot-audio silence ends a captured response:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --silence-timeout 1.1
```

## Caller Voice Options

Caller speech is synthesized with macOS `say`.

Set the speaking rate:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --caller-rate 175
```

Set a specific macOS voice:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --caller-voice Samantha
```

## Eval Date and Turn Limit

The eval date affects relative-date scenarios such as "this Friday" or "next
Monday". By default, evals use May 30, 2026.

Override the eval date:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --eval-date 2026-05-30
```

Limit the number of caller/agent turns per scenario:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --max-turns 6
```

## Output Files

The latest batch writes three top-level outputs:

```text
batch_results.json
batch_runs.jsonl
batch_metrics.csv
```

Detailed artifacts are stored per bot and per scenario:

```text
batch_artifacts/
  <batch_id>/
    <bot_id>/
      <scenario_id>/
        results.json
        runs.jsonl
        latency.jsonl
        bot.log
        eval.log
```

`results.json` is the single-scenario output from `eval_runner.py`.

`runs.jsonl` is the single-scenario trend row from `eval_runner.py`.

`latency.jsonl` contains raw voice latency events emitted by the bot when it
supports `VOICE_LATENCY_LOG_PATH`.

`bot.log` contains the bot process command and combined stdout/stderr.

`eval.log` contains the `eval_runner.py` command and combined stdout/stderr.

## Batch Summary: `batch_results.json`

`batch_results.json` is overwritten on each batch run. It contains:

- `batch_id`, timestamp, bot count, scenario count, task count, and wall time.
- `bot_order` and `scenario_order`.
- One nested summary per bot under `bots`.

Each bot summary includes:

- Bot id, name, passed path, resolved path, and file SHA256.
- Scenario count, passed count, and pass rate.
- Judge error count.
- Infrastructure failure count.
- Voice latency summary.
- Artifact directories and artifact paths.
- Full per-scenario results.

Inspect the summary:

```bash
cat batch_results.json
```

With `jq`:

```bash
jq '.bots | to_entries[] | {bot: .key, pass_rate: .value.pass_rate, passed: .value.passed_count, total: .value.scenario_count}' batch_results.json
```

Show failed scenarios for one bot:

```bash
jq '.bots.bot0.scenarios[] | select(.passed == false) | {id, reason}' batch_results.json
```

## Trend Rows: `batch_runs.jsonl`

`batch_runs.jsonl` appends one aggregate row per bot per batch. This is useful for
plotting bot-level trends over time.

Each row includes:

- Batch id and timestamp.
- Bot metadata and SHA256.
- Scenario count, passed count, and pass rate.
- Judge error count and infrastructure failure count.
- Voice latency summary.
- Wall time.
- Artifact paths.

View the latest rows:

```bash
tail -n 5 batch_runs.jsonl
```

## Plotting Data: `batch_metrics.csv`

`batch_metrics.csv` appends one long-form row per bot per batch for spreadsheets
or plotting tools.

Columns include:

- `batch_id`
- `timestamp`
- `timestamp_iso`
- `bot_id`
- `bot_name`
- `bot_path`
- `bot_resolved_path`
- `bot_sha256`
- `pass_rate`
- `passed_count`
- `scenario_count`
- `judge_error_count`
- `infrastructure_failure_count`
- `ttfa_p50_ms`
- `ttfa_p95_ms`
- `ttla_p50_ms`
- `ttla_p95_ms`
- `completed_voice_turns`
- `wall_time_s`
- `artifact_dirs`
- `artifact_paths`

Open it directly in a spreadsheet, or inspect it from the terminal:

```bash
column -s, -t < batch_metrics.csv | less -S
```

## Custom Output Locations

Set a stable batch id when you want predictable artifact paths:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --batch-id local-comparison-001
```

Write outputs to a custom directory:

```bash
mkdir -p eval-output

uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --batch-id local-comparison-001 \
  --artifacts-root eval-output/artifacts \
  --batch-results-path eval-output/batch_results.json \
  --batch-runs-path eval-output/batch_runs.jsonl \
  --batch-metrics-path eval-output/batch_metrics.csv
```

## How Pass and Failure Are Counted

A scenario passes when the needed answer, action, confirmation, or safe refusal
appears anywhere in the observable agent response. The judge uses a lenient
answer-present policy:

- It passes if the answer is in the response, even if the wording is awkward.
- It accepts semantically equivalent wording.
- It tolerates likely ASR artifacts.
- It accepts spoken confirmation IDs.
- It accepts a different conversational order if the answer/action appears.
- It accepts extra questions or minor omissions around an otherwise correct answer.
- It does not fail voice-mode runs only because tool calls are unavailable.

A scenario still fails when the needed answer/action/refusal is absent or materially
bad:

- Wrong date, time, provider, or confirmation.
- Unsafe medical advice.
- False certainty about unsupported facts.
- No answer/action/refusal that satisfies the scenario.
- Blank transcript.
- Contradiction of a required outcome.

If the judge response cannot be parsed as JSON after one repair attempt, the
scenario is marked:

```json
{
  "passed": false,
  "judge_error": true
}
```

Judge errors count as non-passing in `pass_rate`, but they are also broken out as
`judge_error_count` so infrastructure and evaluator failures can be separated from
agent behavior.

## Infrastructure Failures

The batch runner continues after startup failures and eval timeouts. A failed task
is recorded as a non-passing scenario with `infrastructure_failure: true`.

Common infrastructure failure types:

- `startup_timeout`: the bot did not report ready at `/status` before the timeout.
- `eval_timeout`: the single scenario exceeded `--eval-timeout`.
- `eval_failed`: `eval_runner.py` exited with a non-zero status.

Start debugging with the per-task logs:

```bash
find batch_artifacts -name bot.log -o -name eval.log
```

Open the failing task's logs:

```bash
cat batch_artifacts/<batch_id>/<bot_id>/<scenario_id>/bot.log
cat batch_artifacts/<batch_id>/<bot_id>/<scenario_id>/eval.log
```

## Recommended Workflow

Start with a one-scenario smoke test:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --limit 1 \
  --max-workers 1
```

If that passes infrastructure checks, run a small targeted set:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --scenario happy_booking_cleaning_next_tuesday \
  --scenario insurance_unknown_cigna \
  --scenario emergency_swelling \
  --max-workers 2
```

Then run the full comparison:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --max-workers 4
```

Review pass rates:

```bash
jq '.bots | to_entries[] | {bot: .key, pass_rate: .value.pass_rate, judge_errors: .value.judge_error_count, infra_failures: .value.infrastructure_failure_count}' batch_results.json
```

Review failures:

```bash
jq '.bots | to_entries[] | {bot: .key, failures: [.value.scenarios[] | select(.passed == false) | {id, reason, judge_error, infrastructure_failure}]}' batch_results.json
```

Review latency:

```bash
jq '.bots | to_entries[] | {bot: .key, latency: .value.voice_latency}' batch_results.json
```

## Troubleshooting

If every task fails with `startup_timeout`, run one bot manually:

```bash
uv run bot0.py --host localhost --port 7860
curl -s http://localhost:7860/status
```

If the bot starts manually but batch mode fails, increase startup timeout:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --startup-timeout 120
```

If ASR fails or transcripts are blank, verify the ASR endpoint:

```bash
echo "$NVIDIA_ASR_URL"
```

Then inspect `eval.log` for WebRTC, ASR, or transcription errors.

If caller audio synthesis fails, confirm macOS `say` is available:

```bash
which say
```

If latency is `null`, the bot may not include the latency logger or may not honor
`VOICE_LATENCY_LOG_PATH`. The batch runner sets that variable for each bot process,
but the bot implementation must write raw latency events to that path.

If `eval_runner.py --bots ...` does not accept an option you need, use
`batch_eval_runner.py` directly. The batch runner is the canonical interface for
multi-agent voice evals.

## Verification Commands

After changing eval code or this guide, run:

```bash
uv run pytest
uv run ruff check
```

When live services are available, run:

```bash
uv run python batch_eval_runner.py \
  --bots bot0.py bot1.py \
  --limit 1 \
  --max-workers 1
```
