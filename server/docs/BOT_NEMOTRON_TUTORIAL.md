# Tutorial: Bright Smile Dental Nemotron Voice Agent

This tutorial explains `bot.py`, the Nemotron-powered Bright Smile Dental front-desk
voice agent. The bot can run locally in a browser over WebRTC, or in production on
Pipecat Cloud with Twilio phone-call audio.

```text
Caller audio
  -> transport input
  -> NVIDIA/Nemotron streaming STT
  -> Pipecat user context aggregator
  -> Nemotron LLM served by vLLM
  -> Gradium TTS
  -> transport output
  -> Pipecat assistant context aggregator
```

The application logic is intentionally small. `bot.py` owns the real-time voice
pipeline and transport selection. `tools.py` owns the dental front-desk prompt,
LLM-callable tool schemas, and the in-memory mock appointment backend.

## What the Agent Does

The assistant is Aria, the front-desk assistant for Bright Smile Dental. The opening
line is:

```text
Thanks for calling Bright Smile Dental, this is Aria. How can I help?
```

The supported front-desk tasks are:

- Check appointment availability for a requested date.
- Book an appointment after collecting name, date, time, and reason.
- Reschedule an appointment when the caller gives an existing confirmation ID.
- Check whether Delta Dental, MetLife, or Aetna are on the accepted insurance list.
- End the call cleanly after saying goodbye.

The agent must not claim unsupported capabilities. It cannot cancel appointments, and
it cannot confirm office hours. If callers ask about clinical symptoms, medication,
diagnosis, or treatment, it should not give advice and should offer to book a visit.
For emergencies such as severe pain, facial swelling, trauma, or bleeding, it should
tell the caller to seek emergency care first, then offer an urgent slot.

## Files Involved

`bot.py` is the Pipecat entrypoint. It configures STT, LLM, TTS, WebRTC transport,
Twilio WebSocket transport, VAD, context aggregation, and pipeline lifecycle.

`tools.py` contains the dental behavior:

- `SYSTEM_PROMPT`: Aria's front-desk policy.
- `TOOLS`: OpenAI-style tool schemas.
- `pipecat_tools_schema()`: converts `TOOLS` into Pipecat `FunctionSchema` objects.
- `register_pipecat_functions(llm)`: registers the actual tool handlers.
- `_BOOKINGS`: in-memory mock appointment state.

`nvidia_stt.py` implements `NVidiaWebSocketSTTService`, a Pipecat STT service for the
NVIDIA streaming ASR endpoint.

`nemotron_llm.py` implements `VLLMOpenAILLMService`, a small subclass of Pipecat's
OpenAI-compatible LLM service. It preserves the existing Nemotron/vLLM setup and keeps
voice latency metrics focused on first spoken content.

`metrics.py` records per-turn voice latency after outbound audio has passed through
`transport.output()`. TTFA is caller stop to first outbound audio frame. TTLA is caller
stop to the last outbound audio frame for a completed LLM response. These timings measure
bot-side audio output acceptance, not physical playback on the caller's device.

`Dockerfile` defines the image Pipecat Cloud builds. `pcc-deploy.toml` defines the
Pipecat Cloud deployment name, secret set, agent profile, Krisp VIVA setting, and
minimum warm agents.

## Local Install

Prerequisites:

- Python 3.11 or newer.
- `uv`.
- A reachable NVIDIA/Nemotron streaming ASR WebSocket endpoint.
- A reachable OpenAI-compatible vLLM endpoint serving Nemotron.
- A Gradium API key for TTS.

From the `server` directory, install dependencies:

```bash
uv sync
```

Create a local environment file:

```bash
cp .env.example .env
```

Fill in at least these values:

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

`ENV=local` disables the Pipecat Cloud Krisp VIVA filter path for local runs. Keep
`NEMOTRON_ENABLE_THINKING=false` for voice unless the vLLM server is configured to
separate reasoning from spoken content.

Run the bot locally:

```bash
uv run bot.py
```

Then open the local WebRTC test page:

```text
http://localhost:7860
```

Start a browser call and try:

- "I need a cleaning next Tuesday."
- "Do you take Delta Dental?"
- "I need to reschedule confirmation BSD1001."
- "I have facial swelling and severe pain."
- "Thanks, goodbye."

## Local Twilio Testing

The same `bot.py` also supports Twilio media streams through Pipecat's WebSocket
runner path. The Twilio path uses `WebSocketRunnerArguments`, parses the Twilio stream
metadata, fetches call metadata from Twilio, and configures `TwilioFrameSerializer`.

Add Twilio credentials to `.env`:

```bash
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
```

When a Twilio call arrives, `bot.py` logs the caller and destination numbers if Twilio
metadata is available. The prompt explicitly says caller ID is for logging only: Aria
must not infer patient identity, claim to recognize the caller, or mention caller
records from the phone number alone.

Twilio media streams use 8 kHz audio, so the WebSocket branch overrides both pipeline
sample rates:

```python
transport_overrides["audio_in_sample_rate"] = 8000
transport_overrides["audio_out_sample_rate"] = 8000
```

## Pipecat Cloud Deployment

This project uses Pipecat Cloud cloud builds: the CLI reads `pcc-deploy.toml`, builds
the local `Dockerfile` in the cloud, and deploys the resulting image. Pipecat's current
deployment docs describe `pipecat cloud deploy` as the standard cloud-build path, with
`pcc-deploy.toml` supplying repeatable deployment settings. The same docs note that
secret sets are injected as environment variables into the agent deployment.

### 1. Authenticate the CLI

Install the Pipecat CLI if needed, then log in:

```bash
uv run pipecat cloud auth login
```

If your shell has a globally installed CLI, `pipecat cloud ...` is fine. The project
dependency also installs the CLI into the `uv` environment, so `uv run pipecat ...`
keeps commands pinned to this repo.

### 2. Check `pcc-deploy.toml`

The deployment config should look like this:

```toml
agent_name = "bright-smile-dental"
secret_set = "bright-smile-dental-secrets"
agent_profile = "agent-1x"

[krisp_viva]
audio_filter = "tel"

[scaling]
min_agents = 1
```

Notes:

- `agent_name` is the Pipecat Cloud deployment name.
- `secret_set` must match the secret set you upload.
- `agent-1x` is the default voice-agent profile.
- `audio_filter = "tel"` matches the phone-call use case.
- `min_agents = 1` keeps one warm agent to avoid cold starts.

### 3. Upload Secrets

Use a production `.env` file with the same runtime variables used locally:

```bash
uv run pipecat cloud secrets set bright-smile-dental-secrets --file .env
```

At minimum, the secret set needs:

```bash
GRADIUM_API_KEY=...
GRADIUM_VOICE_ID=...
NVIDIA_ASR_URL=ws://...
NEMOTRON_LLM_URL=http://.../v1
NEMOTRON_LLM_MODEL=nvidia/nemotron-3-super
NEMOTRON_LLM_API_KEY=EMPTY
NEMOTRON_ENABLE_THINKING=false
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
```

Do not commit real secrets. Pipecat Cloud stores them in the named secret set and
injects them into the agent container at runtime.

### 4. Deploy

From the same directory as `pcc-deploy.toml`:

```bash
uv run pipecat cloud deploy
```

The CLI uses this repo's `Dockerfile`. Because `bot.py` imports `tools.py`, the
Dockerfile must copy `tools.py` into the image along with `bot.py`, `nemotron_llm.py`,
and `nvidia_stt.py`.

For a non-interactive deployment, use:

```bash
uv run pipecat cloud deploy --yes
```

### 5. Start a WebRTC Session

After deployment, start a Daily WebRTC session for browser testing:

```bash
uv run pipecat cloud agent start bright-smile-dental --use-daily
```

The command returns session details, including the Daily room URL or session metadata
needed by your frontend.

### 6. Connect Twilio

For phone calls, configure Twilio to stream call audio to the Pipecat Cloud WebSocket
endpoint for the deployed agent. The app code already handles Twilio's WebSocket
runner path:

1. Pipecat receives `WebSocketRunnerArguments`.
2. `parse_telephony_websocket(...)` extracts Twilio stream and call IDs.
3. `get_call_info(...)` optionally fetches the caller and destination numbers.
4. `TwilioFrameSerializer` converts between Twilio media-stream messages and Pipecat
   audio frames.
5. `run_bot(...)` starts with 8 kHz input and output sample rates.

Keep the Twilio credentials in the Pipecat Cloud secret set so `get_call_info(...)`
can call Twilio's REST API.

## Tool Schema and Registration

`tools.py` keeps a single OpenAI-style `TOOLS` list:

```python
TOOLS = [
    {"type": "function", "function": {"name": "check_availability", ...}},
    {"type": "function", "function": {"name": "book_appointment", ...}},
    {"type": "function", "function": {"name": "reschedule_appointment", ...}},
    {"type": "function", "function": {"name": "check_insurance", ...}},
    {"type": "function", "function": {"name": "end_call", ...}},
]
```

`pipecat_tools_schema()` converts each tool into a Pipecat `FunctionSchema`:

```python
return ToolsSchema(
    standard_tools=[_function_schema_from_openai_tool(tool) for tool in TOOLS]
)
```

`register_pipecat_functions(llm)` wires the model-visible tool names to their Python
handlers:

```python
llm.register_function("check_availability", ...)
llm.register_function("book_appointment", ...)
llm.register_function("reschedule_appointment", ...)
llm.register_function("check_insurance", ...)
llm.register_function("end_call", _end_call)
```

`bot.py` uses both pieces:

```python
tools = pipecat_tools_schema()
...
register_pipecat_functions(llm)
context = LLMContext(tools=tools)
```

Both are required. The schema tells the LLM what it may call; registration provides
the actual Python functions Pipecat invokes.

## Dental Tools

### `check_availability`

Input:

```json
{"date": "2026-06-02"}
```

Output:

```json
{"date": "2026-06-02", "open_slots": ["1:00 PM", "2:30 PM", "4:00 PM"]}
```

The mock backend always returns afternoon slots. A real integration would replace
this with a scheduler lookup.

### `book_appointment`

Input:

```json
{
  "name": "Sam Lee",
  "date": "2026-06-02",
  "time": "1:00 PM",
  "reason": "cleaning"
}
```

Output:

```json
{
  "confirmation_id": "BSD1001",
  "status": "booked",
  "name": "Sam Lee",
  "date": "2026-06-02",
  "time": "1:00 PM",
  "reason": "cleaning"
}
```

Aria should only confirm an appointment after this tool returns a confirmation ID.

### `reschedule_appointment`

Input:

```json
{"confirmation_id": "BSD1001", "date": "2026-06-03", "time": "2:30 PM"}
```

If the confirmation ID exists in `_BOOKINGS`, the tool updates that mock booking. If
not, it returns:

```json
{"status": "not_found", "confirmation_id": "BSD1001"}
```

### `check_insurance`

Input:

```json
{"provider": "Delta Dental"}
```

Output:

```json
{"provider": "Delta Dental", "accepted": true, "known_list_only": true}
```

Only Delta Dental, MetLife, and Aetna are known accepted providers. Unknown plans
return `accepted: false`; Aria should say the office will confirm rather than inventing
coverage.

### `end_call`

`end_call` pushes `EndTaskFrame` upstream and returns:

```python
FunctionCallResultProperties(run_llm=False)
```

That prevents a second LLM response after the goodbye. The prompt requires Aria to say
goodbye and call `end_call` in the same turn.

## Prompt Policy

`SYSTEM_PROMPT` is part of the product behavior. It tells Aria to:

- Keep phone replies to one or two short sentences.
- Ask one thing at a time.
- Avoid unsupported cancellation and office-hours claims.
- Avoid clinical, dental, diagnosis, treatment, and medication advice.
- Treat emergencies differently from routine scheduling.
- Confirm bookings only after a confirmation ID is returned.
- Never invent insurance coverage.
- Say goodbye and call `end_call` in the same turn.

`bot.py` adds runtime context:

- Today's date, so relative dates like "next Tuesday" can be interpreted.
- Caller-ID privacy rules for Twilio calls.
- Spoken-output style guidance.

## Voice Pipeline

`run_bot(...)` builds the pipeline in this order:

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

The order matters:

- `transport.input()` receives audio frames from WebRTC or Twilio.
- `stt` converts caller speech to text.
- `user_aggregator` builds user turns and filters incomplete turns.
- `llm` generates spoken text or tool calls.
- `tts` converts assistant text to audio.
- `transport.output()` sends audio back to the browser or phone call.
- `latency_logger` records TTFA and TTLA once audio has been accepted by transport output.
- `assistant_aggregator` records assistant messages in context.

`PipelineParams` enables metrics and uses WebRTC-friendly defaults:

```python
audio_in_sample_rate=16000
audio_out_sample_rate=24000
```

The Twilio branch overrides both to `8000`.

## Transport Selection

`bot(runner_args)` chooses transport by argument type:

- `SmallWebRTCRunnerArguments`: local browser WebRTC.
- `WebSocketRunnerArguments`: Twilio telephony over WebSocket.

For local WebRTC, the transport is:

```python
SmallWebRTCTransport(
    webrtc_connection=webrtc_connection,
    params=TransportParams(
        audio_in_enabled=True,
        audio_in_filter=krisp_filter,
        audio_out_enabled=True,
    ),
)
```

For Twilio, the transport is:

```python
FastAPIWebsocketTransport(
    websocket=runner_args.websocket,
    params=FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_in_filter=krisp_filter,
        audio_out_enabled=True,
        add_wav_header=False,
        serializer=serializer,
    ),
)
```

`add_wav_header=False` matters because Twilio expects raw streaming media frames, not
WAV files.

## Validation Checklist

Run these before deploying:

```bash
uv run ruff check bot.py tools.py
uv run pyright bot.py tools.py
uv run python -c "import bot, tools"
```

Smoke-test the tool backend:

```bash
uv run python - <<'PY'
import tools

print([tool.name for tool in tools.pipecat_tools_schema().standard_tools])
booking = tools.TOOL_IMPLS["book_appointment"](
    {
        "name": "Sam Lee",
        "date": "2026-06-02",
        "time": "1:00 PM",
        "reason": "cleaning",
    }
)
print(booking)
print(
    tools.TOOL_IMPLS["reschedule_appointment"](
        {
            "confirmation_id": booking["confirmation_id"],
            "date": "2026-06-03",
            "time": "2:30 PM",
        }
    )
)
print(tools.TOOL_IMPLS["check_insurance"]({"provider": "Delta Dental"}))
print(tools.TOOL_IMPLS["check_insurance"]({"provider": "Unknown Plan"}))
PY
```

Manual voice checks:

- New appointment: availability lookup, booking, confirmation ID.
- Reschedule: existing confirmation ID moves to a new date and time.
- Known insurance: Delta Dental, MetLife, or Aetna is accepted.
- Unknown insurance: Aria does not invent coverage.
- Clinical question: Aria offers to book a visit without giving advice.
- Emergency: Aria tells the caller to seek emergency care first, then offers an urgent
  slot.
- Goodbye: Aria says goodbye and calls `end_call`.

## Production Upgrade Points

The current backend is a mock. Before using this for a real dental practice, replace
or harden:

- `_BOOKINGS` with a scheduler, PMS, or database integration.
- Insurance checks with a verified eligibility workflow.
- Confirmation IDs with IDs from the real scheduler.
- Emergency language with practice-approved policy.
- Audit logging and PHI handling.
- Twilio webhook authentication and deployment-level WebSocket authentication.
- Monitoring for failed STT, LLM, TTS, and telephony sessions.

The important architectural boundary is already in place: `bot.py` handles real-time
voice infrastructure, while `tools.py` owns the dental domain actions.
