# Tutorial: Bright Smile Dental Nemotron Voice Agent

This tutorial explains the current `bot.py` voice agent. The bot acts as Aria, the front-desk assistant for Bright Smile Dental. Callers can use it through the local browser test page or through a real Twilio phone number connected to Pipecat Cloud.

The runtime pipeline is:

```text
Caller audio
  -> transport input, either SmallWebRTC or Twilio WebSocket
  -> NVIDIA/Nemotron streaming STT
  -> Pipecat user context aggregator
  -> Nemotron LLM served by vLLM
  -> dental tools from tools.py
  -> Gradium TTS
  -> transport output
  -> Pipecat assistant context aggregator
```

The dental backend is intentionally mocked. Appointment data is stored in memory, so it resets when the process restarts. This is useful for demos and testing, but it is not production scheduling infrastructure or HIPAA-compliant patient record storage.

## Files Involved

- `bot.py`: main Pipecat entrypoint. It configures STT, LLM, TTS, tools, local WebRTC, and Twilio telephony.
- `tools.py`: Bright Smile Dental prompt, mock appointment backend, OpenAI-style tool schemas, Pipecat schema conversion, and tool registration.
- `nvidia_stt.py`: custom Pipecat STT service for NVIDIA streaming ASR.
- `nemotron_llm.py`: OpenAI-compatible vLLM service wrapper with corrected TTFB metrics for Nemotron thinking streams.
- `Dockerfile`: Pipecat Cloud image definition. It copies `bot.py`, `tools.py`, `nvidia_stt.py`, and `nemotron_llm.py`.
- `pcc-deploy.toml`: Pipecat Cloud deployment config for `dental-bot`.

## Dental Use Cases

The bot is optimized for short phone calls with one question per turn.

Core use cases:

- Book an appointment after collecting name, date, time, and reason.
- Check open slots before offering appointment times.
- Reschedule an existing appointment using a confirmation id.
- Cancel an existing appointment using a confirmation id.
- Answer office hours: Monday through Friday, 8:00 AM to 5:00 PM.
- Check whether an insurance provider is in the known accepted list: Delta Dental, MetLife, and Aetna.
- Deflect medical advice. For clinical questions, it offers to book a visit.
- Deflect emergencies. For severe pain, facial swelling, trauma, or bleeding, it tells the caller to call 911 or go to the ER now, then offers dental follow-up.

Example test calls:

```text
"I'd like to book a cleaning next Tuesday."
"Do you take Delta Dental?"
"I need to reschedule confirmation BSD1001."
"Cancel appointment BSD1001."
"What are your hours?"
"My face is swollen and my tooth hurts badly."
```

## Tool Model

`tools.py` is the single source of truth for LLM-callable tools.

The public tool names are:

- `check_availability`
- `book_appointment`
- `reschedule_appointment`
- `cancel_appointment`
- `check_insurance`
- `end_call`

`TOOLS` stores OpenAI-style function declarations. `build_pipecat_tools_schema()` converts those declarations into Pipecat `FunctionSchema` objects because this Pipecat version expects `ToolsSchema(standard_tools=...)` to receive Pipecat schemas or direct callables.

`register_pipecat_functions(llm)` registers every handler with:

```python
llm.register_function(name, handler)
```

The handler reads `params.arguments`, calls the matching function in `TOOL_IMPLS`, and returns the result with:

```python
await params.result_callback(result)
```

`end_call` is special. It pushes an `EndTaskFrame` upstream and returns `FunctionCallResultProperties(run_llm=False)` so the bot can say goodbye and then hang up without asking the LLM for another turn.

## Local Server: Step by Step

Run from the `server` directory.

1. Install `uv` if needed.

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install project dependencies.

   ```bash
   cd /Users/pmui/dev/ideas/kittie/dental-voice-agent/server
   uv sync
   ```

3. Create or update `.env`.

   ```bash
   touch .env
   ```

   Minimum local variables:

   ```bash
   GRADIUM_API_KEY=...
   ENV=local
   ```

   Nemotron variables are optional if the defaults point at a reachable local or hackathon endpoint:

   ```bash
   NVIDIA_ASR_URL=ws://192.168.7.228:8081
   NEMOTRON_LLM_URL=http://192.168.7.228:8000/v1
   NEMOTRON_LLM_MODEL=nvidia/nemotron-3-super
   NEMOTRON_LLM_API_KEY=EMPTY
   NEMOTRON_ENABLE_THINKING=false
   GRADIUM_VOICE_ID=Eu9iL_CYe8N-Gkx_
   ```

   Keep `NEMOTRON_ENABLE_THINKING=false` for voice unless the vLLM server is configured to keep reasoning out of streamed spoken content.

4. Start the local Pipecat server.

   ```bash
   uv run bot.py
   ```

5. Open the local browser client.

   ```text
   http://localhost:7860
   ```

6. Click Connect and speak to the bot.

   A good first test is:

   ```text
   I want to book a cleaning next Tuesday.
   ```

7. Stop the local server with Ctrl-C.

## Expected Local Conversation Flow

1. Browser connects to `localhost:7860`.
2. Pipecat calls `bot(runner_args)` with `SmallWebRTCRunnerArguments`.
3. `bot.py` creates a `SmallWebRTCTransport`.
4. `run_bot(...)` builds the dental system instruction.
5. `run_bot(...)` creates STT, LLM, TTS, tool schema, and tool registrations.
6. The `on_client_connected` handler injects the greeting request.
7. The LLM says: "Thanks for calling Bright Smile Dental, this is Aria. How can I help?"
8. When the caller asks for appointment times, the LLM calls `check_availability`.
9. When the caller confirms details, the LLM calls `book_appointment`.
10. The tool returns a confirmation id like `BSD1001`.
11. The LLM speaks the confirmed appointment details.
12. When the caller is done, the LLM says goodbye and calls `end_call`.

## Deploy to Pipecat Cloud

The deployment path uses Pipecat Cloud for the running bot and Twilio for the public phone number.

### 1. Install and authenticate the Pipecat CLI

```bash
uv tool install pipecat-ai-cli
pc cloud auth login
```

Confirm the CLI can see your organizations:

```bash
pc cloud organizations list
```

### 2. Review deployment config

`pcc-deploy.toml` should contain:

```toml
agent_name = "dental-bot"
secret_set = "dental-bot-secrets"
agent_profile = "agent-1x"

[krisp_viva]
	audio_filter = "tel"

[scaling]
	min_agents = 1
```

### 3. Upload secrets

Make sure `.env` contains the runtime secrets needed in cloud:

```bash
GRADIUM_API_KEY=...
NVIDIA_ASR_URL=...
NEMOTRON_LLM_URL=...
NEMOTRON_LLM_MODEL=nvidia/nemotron-3-super
NEMOTRON_LLM_API_KEY=EMPTY
NEMOTRON_ENABLE_THINKING=false
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
```

Upload them:

```bash
pc cloud secrets set dental-bot-secrets --file .env
```

Confirm the secret set exists:

```bash
pc cloud secrets list
```

### 4. Deploy

From the `server` directory:

```bash
pc cloud deploy
```

If you need to redeploy even when the CLI thinks nothing changed:

```bash
pc cloud deploy --force
```

Check the deployed agent:

```bash
pc cloud agent status dental-bot
pc cloud agent deployments dental-bot
```

### 5. Configure Twilio

1. Buy or select a Twilio phone number with voice capability.
2. Create a TwiML Bin with this content:

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <Response>
     <Connect>
       <Stream url="wss://api.pipecat.daily.co/ws/twilio">
         <Parameter name="_pipecatCloudServiceHost"
           value="dental-bot.YOUR_ORG_NAME"/>
       </Stream>
     </Connect>
   </Response>
   ```

3. Replace `YOUR_ORG_NAME` with the organization name from `pc cloud organizations list`.
4. In Twilio, attach the TwiML Bin to the phone number under Voice Configuration.
5. Call the number and verify the Bright Smile Dental greeting.

## Remote Phone Call Flow

1. Caller dials the Twilio number.
2. Twilio runs the TwiML Bin.
3. Twilio opens a media stream to `wss://api.pipecat.daily.co/ws/twilio`.
4. Pipecat Cloud routes the stream to `dental-bot.YOUR_ORG_NAME`.
5. `bot.py` receives `WebSocketRunnerArguments`.
6. The bot parses Twilio stream metadata with `parse_telephony_websocket`.
7. The bot fetches caller/callee metadata from Twilio using `get_call_info`.
8. The bot creates a `TwilioFrameSerializer` and `FastAPIWebsocketTransport`.
9. The pipeline runs at 8 kHz input and 8 kHz output because Twilio media streams use telephony audio.

## Debug Locally

Start with the narrowest check that matches the failure.

### Code and tool checks

```bash
uv run python -m compileall bot.py tools.py
uv run pytest
uv run ruff check bot.py tools.py test_tools.py
```

The full repo currently has an unrelated `ruff` import-order issue in `metrics.py`, so use the targeted ruff command above when validating only the dental bot change.

### The server does not start

Check:

- You are in the `server` directory.
- `uv sync` has completed.
- `.env` contains `GRADIUM_API_KEY`.
- `ENV=local` is set for local testing so the bot does not try to import the cloud Krisp filter.

Useful command:

```bash
ENV=local uv run bot.py
```

### Browser connects but the bot never speaks

Check:

- Browser microphone permission is allowed.
- The terminal shows `Client connected`.
- `GRADIUM_API_KEY` is valid.
- The configured `NEMOTRON_LLM_URL` is reachable.
- The configured `NVIDIA_ASR_URL` is reachable.

Use simple import checks:

```bash
uv run python -c "import bot, tools; print('imports ok')"
uv run python -c "from tools import build_pipecat_tools_schema; print([t.name for t in build_pipecat_tools_schema().standard_tools])"
```

### The bot hears poorly or repeats partial words

Check:

- `NVIDIA_ASR_URL` points to the expected streaming ASR service.
- Browser microphone input is clean.
- `strip_interim_prefix=True` is still enabled in `bot.py`.
- You are testing through the local browser for WebRTC and through Twilio only for phone audio.

### Tool calls do not work

Run the tests:

```bash
uv run pytest test_tools.py
```

Then inspect:

- `TOOLS` includes the public function schema.
- `TOOL_IMPLS` includes the same name.
- `register_pipecat_functions(llm)` registers every name.
- `build_pipecat_tools_schema()` returns all expected tool names.

### The assistant gives medical advice

Tighten `SYSTEM_PROMPT` in `tools.py` and the "Voice behavior" block in `bot.py`. The intended behavior is to avoid clinical guidance, offer scheduling for routine clinical questions, and route emergency symptoms to 911 or the ER.

### The assistant speaks reasoning out loud

Set:

```bash
NEMOTRON_ENABLE_THINKING=false
```

Reasoning should stay off for voice unless the vLLM server is confirmed to emit reasoning into a separate non-spoken field.

## Debug Remotely

Remote debugging has three layers: cloud deployment, cloud runtime logs, and Twilio call routing.

### Check deployment status

```bash
pc cloud agent status dental-bot
pc cloud agent deployments dental-bot
```

If a cloud build failed, list builds and inspect logs:

```bash
pc cloud build list
pc cloud build logs BUILD_ID --limit 1000
```

### Check live sessions

```bash
pc cloud agent sessions dental-bot
```

If you see a session id, filter logs to that session:

```bash
pc cloud agent logs dental-bot --session-id SESSION_ID --limit 300
```

For general logs:

```bash
pc cloud agent logs dental-bot --limit 300
pc cloud agent logs dental-bot --level ERROR --limit 100
```

### Common remote failures

No call reaches the bot:

- Twilio number is not attached to the TwiML Bin.
- TwiML Bin still points at an old service name instead of `dental-bot.YOUR_ORG_NAME`.
- Organization name in `_pipecatCloudServiceHost` is wrong.
- `dental-bot` is not deployed or has zero warm agents.

Call connects, then immediately hangs up:

- Cloud logs show an import error.
- `Dockerfile` is missing a copied source file.
- Required secrets are absent from `dental-bot-secrets`.
- `GRADIUM_API_KEY`, `NVIDIA_ASR_URL`, or `NEMOTRON_LLM_URL` is invalid.

Caller audio reaches the bot, but no caller id is logged:

- `TWILIO_ACCOUNT_SID` or `TWILIO_AUTH_TOKEN` is missing.
- Twilio REST API call failed.
- This does not block the bot; caller lookup is optional.

Bot works locally but not by phone:

- Confirm Twilio path uses 8 kHz sample-rate overrides in `bot.py`.
- Confirm `TwilioFrameSerializer` is receiving `stream_sid` and `call_sid`.
- Check Pipecat Cloud logs while placing a live call.

### Update cloud secrets after changing `.env`

```bash
pc cloud secrets set dental-bot-secrets --file .env
pc cloud deploy --force
```

### Restart by redeploying

For this project, the simplest restart path is:

```bash
pc cloud deploy --force
```

Then place another phone call and check:

```bash
pc cloud agent sessions dental-bot
pc cloud agent logs dental-bot --limit 300
```

## Customizing the Dental Agent

### Change office hours

Edit `OFFICE_HOURS` in `tools.py`.

### Change accepted insurance

Edit both:

- `ACCEPTED_INSURANCE`
- The accepted-insurance sentence in `SYSTEM_PROMPT`

Keep these aligned so the bot does not promise coverage that the tool rejects.

### Add a new appointment workflow

Add the behavior in three places in `tools.py`:

1. Add a backend function, such as `_lookup_appointment`.
2. Add it to `TOOL_IMPLS`.
3. Add its OpenAI-style function schema to `TOOLS`.

Then add or update tests in `test_tools.py`.

### Replace the mock backend

Replace the internals of the `_book_appointment`, `_reschedule_appointment`, and `_cancel_appointment` functions with calls to your real scheduler or CRM. Keep the tool return shapes stable so the LLM can continue using the same prompt and schemas.

## Mental Model

`bot.py` is the real-time audio pipeline. `tools.py` is the dental business layer. Pipecat handles transport, frames, turn detection, tool-call plumbing, and model service integration. Nemotron decides whether to answer directly or call a tool. The tool result is returned to Nemotron, then Gradium speaks the response back to the caller.

For local debugging, look at process logs and run unit tests. For phone debugging, look at Pipecat Cloud agent logs, active sessions, build logs, and Twilio routing.
