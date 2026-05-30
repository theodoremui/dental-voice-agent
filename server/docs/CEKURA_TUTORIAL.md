# Cekura Evaluation Harness Tutorial

This tutorial explains how this repo uses Cekura to evaluate the Bright Smile
Dental voice agent scenarios in `scenarios.py`.

The short version:

- `scenarios.py` remains the source of truth for the 25 canonical scenarios.
- `cekura_eval_runner.py` converts those scenarios into Cekura evaluator payloads.
- One shared Cekura LLM Judge metric checks each scenario against its own success
  criteria.
- Existing local eval transcripts can also be uploaded to Cekura Observability as
  call logs, so Cekura can judge and track runs produced by `eval_runner.py` or
  `batch_eval_runner.py`.

## Why This Is The Simplest Integration

The repo already has a complete scenario set:

```python
SCENARIOS = [
    {
        "id": "...",
        "category": "...",
        "severity": "...",
        "persona": "...",
        "criteria": "...",
    },
]
```

The Cekura harness does not duplicate those scenarios. It maps each local scenario
directly:

- `id` becomes a stable Cekura tag: `scenario:<id>`.
- `category` becomes a Cekura tag: `category:<category>`.
- `severity` becomes a Cekura tag: `severity:<severity>`.
- `persona` becomes the Cekura evaluator instructions for the Testing Agent.
- `criteria` becomes the Cekura expected outcome and is embedded into the
  instructions.

This gives Cekura enough information to simulate the caller and judge the main
agent, while keeping scenario edits in one place.

## Key Concepts

### Cekura Project

A project is the workspace container for related agents, evaluators, metrics, and
results. Use one project for this dental voice-agent evaluation suite unless you
need separate environments.

### Cekura Agent

A Cekura agent is the agent under test. It stores provider configuration, phone or
chat/websocket settings, prompt metadata, and links to metrics and evaluators. For
this repo, the Cekura agent represents the Bright Smile Dental voice agent.

### Assistant ID

The assistant ID is the provider-side identifier for the agent. Depending on the
integration, this may be a Retell, Vapi, ElevenLabs, Bland, or self-hosted
assistant identifier. The harness accepts `--assistant-id` where Cekura supports it.

### Evaluator

An evaluator is a Cekura test scenario. It describes the Testing Agent's caller
role, behavior, and expected outcome. In this repo, every entry in `SCENARIOS`
becomes one Cekura evaluator.

### Testing Agent

The Testing Agent is Cekura's simulated caller. It plays the patient or caller from
the local scenario `persona`.

### Main Agent

The Main Agent is the dental voice agent being tested. Metrics judge whether the
Main Agent satisfied the scenario criteria.

### Personality

A personality controls how the Testing Agent behaves and sounds: language, accent,
interruptiveness, speed, background noise, voice model, and related caller behavior.
Every Cekura evaluator needs a personality ID.

### Metric

A metric is the grading rule. Cekura supports predefined metrics, LLM Judge
metrics, and code metrics. This harness uses one shared LLM Judge metric, because
the local scenario `criteria` already contains the scenario-specific pass/fail
rule.

### LLM Judge Metric

An LLM Judge metric evaluates a transcript with a natural-language prompt. The
shared metric created by this harness checks the transcript against either:

- `{{evaluator.instructions}}` for Cekura simulation runs.
- `{{dynamic_variables.criteria}}` for uploaded local eval transcripts.

### Metric Variables

Metric variables are placeholders Cekura resolves when a metric runs. Important
ones for this harness:

- `{{transcript}}`: full conversation transcript.
- `{{evaluator.instructions}}`: scenario instructions during Cekura simulation.
- `{{dynamic_variables.criteria}}`: local scenario criteria when uploading a local
  transcript through Observability.
- `{{dynamic_variables.scenario_id}}`, `category`, `severity`, and `persona`: extra
  context added by `cekura_eval_runner.py`.

### Simulation

Simulation means Cekura runs the evaluator against the configured agent. Use this
when Cekura can reach the agent through a supported chat, voice, websocket, or
provider integration.

### Observability

Observability means sending existing call logs or transcripts to Cekura. Use this
when the local harness already produced transcripts and you want Cekura to store,
judge, and compare them.

### Call Log

A call log is an observed conversation stored in Cekura. `ingest-results` converts
local `results.json` or `batch_results.json` transcripts into Cekura call logs.

### Transcript Format

For uploaded local transcripts, the harness uses Cekura transcript format:

```json
[
  {
    "role": "Testing Agent",
    "content": "I need a cleaning.",
    "start_time": 0.0,
    "end_time": 1.2
  },
  {
    "role": "Main Agent",
    "content": "I can help with that.",
    "start_time": 1.4,
    "end_time": 2.6
  }
]
```

Cekura expects the role names `Testing Agent` and `Main Agent` for this format.
The local runner uses `caller` and `agent`, so `cekura_eval_runner.py` translates
the roles and assigns synthetic monotonic timestamps.

### Result

A result is a Cekura test execution group. Running several evaluators together
returns one result ID.

### Run

A run is one evaluator execution inside a result. If you run 25 scenarios once,
the result contains 25 runs.

### Tags

Tags are stable labels used by the harness for idempotency and filtering. The most
important tag is `scenario:<local_scenario_id>`. Re-running `sync` uses that tag
to avoid creating duplicate evaluators.

### Folder

A folder organizes evaluators in Cekura. This harness does not set a folder by
default because Cekura requires folders to exist before evaluators can be assigned
to them. Use `--folder-path` only after creating the folder in Cekura.

### API Key

The harness authenticates with the `CEKURA_API_KEY` value in `.env`. Do not print
or commit the key.

## Prerequisites

Install project dependencies:

```bash
uv sync
```

Confirm `.env` contains:

```bash
CEKURA_API_KEY=...
```

You also need these Cekura IDs from the dashboard or API:

- `project_id`: Cekura project ID.
- `agent_id`: Cekura agent ID for the Bright Smile Dental agent.
- `assistant_id`: provider assistant ID if creating the shared metric or using
  assistant-based APIs.
- `personality_id`: Cekura personality ID for the Testing Agent.

List available English personalities:

```bash
uv run python cekura_eval_runner.py list-personalities --language en
```

Optionally filter by project:

```bash
uv run python cekura_eval_runner.py list-personalities \
  --project-id 123 \
  --language en
```

## Step 1: Create The Shared Metric

Create one LLM Judge metric for all 25 local scenarios:

```bash
uv run python cekura_eval_runner.py create-metric \
  --project-id 123 \
  --assistant-id asst_bright_smile \
  --agent-id 2142 \
  --output cekura_metric.json
```

Inspect `cekura_metric.json` and copy the returned metric `id`.

Dry-run the metric payload first if you want to inspect it without calling Cekura:

```bash
uv run python cekura_eval_runner.py create-metric \
  --project-id 123 \
  --assistant-id asst_bright_smile \
  --agent-id 2142 \
  --dry-run
```

What this does:

- Creates a project-level `llm_judge` metric.
- Enables it for simulation and observability.
- Uses `binary_workflow_adherence`, which is a pass/fail workflow-style score.
- Reads scenario-specific criteria through Cekura metric variables.

## Step 2: Preview Evaluator Payloads

Preview one evaluator:

```bash
uv run python cekura_eval_runner.py plan \
  --agent-id 2142 \
  --personality-id 42 \
  --metric-id 501 \
  --limit 1
```

Preview all 25 and save them:

```bash
uv run python cekura_eval_runner.py plan \
  --agent-id 2142 \
  --personality-id 42 \
  --metric-id 501 \
  --output cekura_plan.json
```

The plan command does not create anything. It shows exactly how each local
scenario will map to Cekura.

## Step 3: Sync The 25 Scenarios To Cekura

Dry-run the sync:

```bash
uv run python cekura_eval_runner.py sync \
  --agent-id 2142 \
  --personality-id 42 \
  --metric-id 501 \
  --dry-run
```

Create missing evaluators:

```bash
uv run python cekura_eval_runner.py sync \
  --agent-id 2142 \
  --personality-id 42 \
  --metric-id 501 \
  --output cekura_sync.json
```

Sync is idempotent. It lists existing evaluators and skips any evaluator with the
matching `scenario:<id>` tag or matching generated name.

Sync only selected scenarios:

```bash
uv run python cekura_eval_runner.py sync \
  --agent-id 2142 \
  --personality-id 42 \
  --metric-id 501 \
  --scenario emergency_swelling \
  --scenario insurance_unknown_cigna
```

## Step 4: Run Cekura Text Simulations

Use text runs when the Cekura agent has a configured chat/text/websocket path:

```bash
uv run python cekura_eval_runner.py run-text \
  --agent-id 2142 \
  --scenario emergency_swelling \
  --scenario insurance_unknown_cigna \
  --frequency 1 \
  --output cekura_text_result.json
```

Run all synced scenarios:

```bash
uv run python cekura_eval_runner.py run-text \
  --agent-id 2142 \
  --output cekura_text_result.json
```

If you already know Cekura evaluator IDs, you can bypass local scenario lookup:

```bash
uv run python cekura_eval_runner.py run-text \
  --agent-id 2142 \
  --evaluator-id 1001,1002,1003
```

If your Cekura setup uses a websocket integration:

```bash
uv run python cekura_eval_runner.py run-text \
  --agent-id 2142 \
  --scenario happy_booking_cleaning_next_tuesday \
  --websocket-url wss://example.com/agent
```

## Step 5: Fetch Results

After `run-text`, Cekura returns a result ID. Fetch it:

```bash
uv run python cekura_eval_runner.py get-result 9876 \
  --output cekura_result_9876.json
```

The result contains run status, transcripts, metric evaluation, and success
summary.

If you have run IDs:

```bash
uv run python cekura_eval_runner.py list-runs 111 112 113 \
  --output cekura_runs.json
```

## Step 6: Upload Local Eval Transcripts To Cekura

This is the best path when you want to keep using the existing local voice harness
and use Cekura for external scoring, storage, and trend analysis.

First produce local results:

```bash
uv run python batch_eval_runner.py \
  --bots bot1.py \
  --limit 1 \
  --max-workers 1
```

Dry-run the upload:

```bash
uv run python cekura_eval_runner.py ingest-results \
  --results-path batch_results.json \
  --agent-id 2142 \
  --metric-id 501 \
  --dry-run
```

Upload to Cekura Observability:

```bash
uv run python cekura_eval_runner.py ingest-results \
  --results-path batch_results.json \
  --agent-id 2142 \
  --metric-id 501 \
  --output cekura_ingest.json
```

Upload only one scenario from a local result file:

```bash
uv run python cekura_eval_runner.py ingest-results \
  --results-path results.json \
  --agent-id 2142 \
  --metric-id 501 \
  --scenario medical_advice_ibuprofen
```

What this does:

- Reads local `results.json` or `batch_results.json`.
- Converts `caller` turns to `Testing Agent`.
- Converts `agent` turns to `Main Agent`.
- Sends `criteria`, `persona`, `category`, `severity`, local verdict, and bot
  metadata as dynamic variables and metadata.
- Passes `metric_ids` so Cekura evaluates the uploaded call.

## Common Workflows

### First-Time Setup

```bash
uv run python cekura_eval_runner.py list-personalities --language en
uv run python cekura_eval_runner.py create-metric --project-id 123 --assistant-id asst_bright_smile --agent-id 2142
uv run python cekura_eval_runner.py sync --agent-id 2142 --personality-id 42 --metric-id 501
```

### Fast CI-Style Local Run Plus Cekura Upload

```bash
uv run python batch_eval_runner.py --bots bot1.py --limit 5 --max-workers 1
uv run python cekura_eval_runner.py ingest-results --results-path batch_results.json --agent-id 2142 --metric-id 501
```

### Cekura-Only Regression Run

```bash
uv run python cekura_eval_runner.py run-text --agent-id 2142 --frequency 1
```

## Troubleshooting

### `CEKURA_API_KEY is required`

Add `CEKURA_API_KEY` to `.env` in the `server` directory.

### Missing Personality ID

Run:

```bash
uv run python cekura_eval_runner.py list-personalities --language en
```

Pick the personality that best matches the caller behavior you want.

### Missing Evaluator IDs During `run-text`

Run `sync` first. `run-text` resolves Cekura evaluator IDs by reading evaluators
tagged with `scenario:<id>`.

### Duplicate Evaluators

The harness skips evaluators with matching `scenario:<id>` tags. If evaluators
were created manually without those tags, either add the tags in Cekura or delete
the duplicates manually.

### Folder Errors

Do not pass `--folder-path` until the folder exists in Cekura. Cekura does not
auto-create evaluator folders from the evaluator create API.

### Uploaded Calls Are Not Evaluated

Check that `--metric-id` was passed to `ingest-results`, and that the metric has
Observability enabled.

### Text Runs Do Not Start

Confirm the Cekura agent has a working chat/text/websocket integration. The local
Pipecat voice runner is not automatically exposed to Cekura.

## Files Added

- `cekura_eval_runner.py`: Cekura REST harness.
- `test_cekura_eval_runner.py`: payload mapping and transcript conversion tests.
- `docs/CEKURA_TUTORIAL.md`: this tutorial.
