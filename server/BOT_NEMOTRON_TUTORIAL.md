# Tutorial: How `bot-nemotron.py` Works

This tutorial explains `bot-nemotron.py`, the Nemotron version of the Field & Flower voice agent. The bot is a Pipecat voice pipeline that lets a caller order flowers by phone or browser:

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

The bot's business logic is intentionally small and hackable. Flower catalog data and known-customer records live in `mock_backend.py`; the LLM calls Python functions in `bot-nemotron.py` as tools; and Pipecat handles streaming audio, turn detection, model calls, speech synthesis, and transport lifecycle.

## Files Involved

`bot-nemotron.py` is the main entrypoint. It defines the ordering tools, builds the prompt, configures STT/LLM/TTS, assembles the pipeline, and selects the correct transport for browser WebRTC or Twilio telephony.

`mock_backend.py` contains in-memory backend data:

- `BOUQUETS`: the flower catalog, including price, description, stock state, occasion tags, and special/deal status.
- `KNOWN_CUSTOMERS`: phone numbers mapped to saved customer data, used only when the Twilio path can identify the caller.

`nvidia_stt.py` implements `NVidiaWebSocketSTTService`, a Pipecat STT service for the NVIDIA streaming ASR endpoint.

`nemotron_llm.py` implements `VLLMOpenAILLMService`, a small subclass of Pipecat's OpenAI-compatible LLM service. Its main purpose is accurate TTFB metrics when Nemotron reasoning is enabled.

## How to Run It

From the `server` directory:

```bash
uv sync
uv run bot-nemotron.py
```

For local browser testing, open:

```text
http://localhost:7860
```

The important environment variables are:

```bash
export GRADIUM_API_KEY=...
export GRADIUM_VOICE_ID=...                  # optional

export NVIDIA_ASR_URL=ws://...               # optional; defaults to local hackathon endpoint

export NEMOTRON_LLM_URL=http://.../v1        # optional; OpenAI-compatible vLLM base URL
export NEMOTRON_LLM_MODEL=nvidia/nemotron-3-super
export NEMOTRON_LLM_API_KEY=EMPTY            # optional; vLLM often ignores this
export NEMOTRON_ENABLE_THINKING=false        # optional; keep false for voice unless configured safely

export TWILIO_ACCOUNT_SID=...                # Twilio path only
export TWILIO_AUTH_TOKEN=...                 # Twilio path only
export ENV=local                             # disables Krisp filter locally
```

`load_dotenv(override=True)` runs near the top of the file, so values in `.env` are loaded into `os.environ` and override existing environment values.

## Top-Level Imports

The imports fall into a few groups:

- Standard library:
  - `os` reads environment variables.
  - `random` generates fake confirmation numbers.
  - `date` injects today's date into the system prompt.
- Network/config/logging:
  - `aiohttp` calls the Twilio REST API.
  - `dotenv.load_dotenv` loads `.env`.
  - `loguru.logger` emits runtime logs.
- Pipecat primitives:
  - `Pipeline`, `PipelineWorker`, `PipelineParams`, and `WorkerRunner` run the streaming graph.
  - `LLMContext`, `LLMContextAggregatorPair`, and `LLMUserAggregatorParams` manage conversational context.
  - `ToolsSchema` describes Python functions as tools the LLM can call.
  - `LLMRunFrame`, `EndTaskFrame`, `FunctionCallResultProperties`, and `FrameDirection` control pipeline events.
  - Transport classes support browser WebRTC and Twilio WebSocket calls.
  - `SileroVADAnalyzer` and `FilterIncompleteUserTurnStrategies` help decide when a caller has finished speaking.
- Project-local services:
  - `BOUQUETS` and `KNOWN_CUSTOMERS` are the mock backend.
  - `VLLMOpenAILLMService` connects to Nemotron through vLLM's OpenAI-compatible API.
  - `NVidiaWebSocketSTTService` connects to NVIDIA streaming STT.

## Function: `get_call_info(call_sid)`

```python
async def get_call_info(call_sid: str) -> dict:
```

This helper fetches Twilio call metadata when the bot is running as a Twilio media-stream endpoint.

### Purpose

The Twilio WebSocket connection gives the bot a call ID. The bot then uses the Twilio REST API to ask, "Who is this call from and what number did they call?" That lets the bot look up `KNOWN_CUSTOMERS` by caller phone number.

### Step-by-step

1. Read `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` from the environment.
2. If either credential is missing, log a warning and return `{}`.
3. Build the Twilio Calls API URL:

   ```text
   https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json
   ```

4. Create `aiohttp.BasicAuth(account_sid, auth_token)`.
5. Open an `aiohttp.ClientSession`.
6. `GET` the call resource.
7. If Twilio returns a non-200 status, log the response body and return `{}`.
8. Parse the JSON response.
9. Return only the fields the bot needs:

   ```python
   {
       "from_number": data.get("from"),
       "to_number": data.get("to"),
   }
   ```

10. If any exception occurs, log the error and return `{}`.

### Why it returns `{}` on failure

Caller lookup is a convenience, not a requirement. If Twilio credentials are missing or the API call fails, the bot still works as a generic new-customer flower shop agent.

## Function: `run_bot(...)`

```python
async def run_bot(
    transport: BaseTransport,
    from_number: str | None = None,
    audio_in_sample_rate: int = 16000,
    audio_out_sample_rate: int = 24000,
):
```

`run_bot` is the core of the application. It does not care whether the caller arrived through WebRTC or Twilio; it receives a configured Pipecat transport and builds the rest of the voice pipeline around it.

### Arguments

- `transport`: the Pipecat transport for input and output audio.
- `from_number`: optional caller phone number, used for known-customer personalization.
- `audio_in_sample_rate`: input audio sample rate passed to `PipelineParams`.
- `audio_out_sample_rate`: output audio sample rate passed to `PipelineParams`.

Defaults are tuned for local WebRTC:

- 16 kHz input audio.
- 24 kHz output audio.

The Twilio path overrides both to 8 kHz because Twilio media streams use 8 kHz telephony audio.

### Per-call state

Inside `run_bot`, this dictionary stores the order:

```python
order: dict = {"items": [], "delivery": None}
```

This is deliberately created inside `run_bot`, not globally. Each call gets a fresh `order` dictionary. The tool functions close over this dictionary, so they can mutate it without using a database or session store.

In production, this is the part you would replace with persistent state: a database row, CRM cart, order API, or workflow engine.

## LLM Tool Functions

The functions nested inside `run_bot` are direct-call tools for the LLM. Each function accepts `params: FunctionCallParams` as its first argument and reports results with:

```python
await params.result_callback(...)
```

That callback sends structured tool output back to the LLM so it can continue the conversation.

The bot uses a two-part tool setup:

```python
tools = ToolsSchema(standard_tools=tool_functions)
...
llm.register_direct_function(fn)
```

`ToolsSchema` describes the tool names, docstrings, and argument schemas to the model. `register_direct_function` wires those model-visible tool names to actual Python handlers.

Both pieces are required.

### Tool: `list_bouquets`

```python
async def list_bouquets(
    params: FunctionCallParams,
    occasion: str | None = None,
    specials_only: bool = False,
) -> None:
```

This tool lists available bouquets from `BOUQUETS`.

The LLM should call it when:

- The caller asks what is available.
- The caller mentions an occasion.
- The caller asks for deals or specials.

The tool has two filters:

- `occasion`: a lowercase occasion like `"birthday"` or `"sympathy"`.
- `specials_only`: `True` to show only bouquets marked as specials.

The filtering logic is:

1. Start with an empty `results` list.
2. Loop through `BOUQUETS.items()`.
3. Skip bouquets where `info["in_stock"]` is false.
4. If `specials_only` is true, skip bouquets where `on_special` is false.
5. If `occasion` is provided:
   - Normalize it with `strip().lower()`.
   - Lowercase the bouquet's occasion tags.
   - Keep the bouquet if the requested occasion appears in a tag or the tag appears in the requested occasion.
6. Append matching bouquets in this shape:

   ```python
   {"name": name, **info}
   ```

The partial-match occasion logic matters. It lets `"mom's birthday"` still match a bouquet tagged `"birthday"` if the LLM passes a close but slightly longer phrase.

If no bouquets match a filtered search, the tool returns:

```python
{
    "bouquets": [],
    "note": "No bouquets match those filters. ..."
}
```

That `note` is guidance for the LLM. It tells the assistant how to recover conversationally: say there is no specific match, then offer the full catalog or another angle.

If no filters were used and no results are found, the tool simply returns an empty `bouquets` list. In the current mock catalog, that should not happen unless all items are out of stock.

### Tool: `check_availability`

```python
async def check_availability(params: FunctionCallParams, bouquet_name: str) -> None:
```

This tool checks whether one specific bouquet can be ordered today.

It normalizes the requested bouquet name:

```python
item = BOUQUETS.get(bouquet_name.lower())
```

Then it returns one of three possible results:

- Unknown bouquet:

  ```python
  {
      "available": False,
      "reason": "We don't carry a bouquet called '...'."
  }
  ```

- Known but sold out:

  ```python
  {
      "available": False,
      "reason": "... is sold out today."
  }
  ```

- Available:

  ```python
  {
      "available": True,
      "price": item["price"]
  }
  ```

This tool is useful when the caller names a bouquet directly and the assistant needs to verify stock before proceeding.

### Tool: `add_to_order`

```python
async def add_to_order(
    params: FunctionCallParams, bouquet_name: str, quantity: int = 1
) -> None:
```

This tool mutates the per-call `order` dictionary by adding a line item.

The docstring tells the LLM an important policy:

```text
Only call this after the customer has confirmed they want this bouquet.
```

The implementation performs the same backend validation as `check_availability`:

1. Lowercase `bouquet_name`.
2. Look it up in `BOUQUETS`.
3. Reject unknown bouquet names.
4. Reject sold-out bouquets.
5. Append an item to `order["items"]`:

   ```python
   {
       "bouquet": bouquet_name.lower(),
       "quantity": quantity,
       "price": item["price"],
   }
   ```

6. Return:

   ```python
   {"ok": True, "items": order["items"]}
   ```

The price is copied into the order line at add time. If the mock catalog changed later during a real call, the current order would still retain the price the customer accepted.

### Tool: `get_order_summary`

```python
async def get_order_summary(params: FunctionCallParams) -> None:
```

This tool reads the current order back to the LLM.

It computes:

```python
total = sum(line["price"] * line["quantity"] for line in order["items"])
```

Then it returns:

```python
{
    "items": order["items"],
    "total": round(total, 2),
    "delivery": order["delivery"],
}
```

The LLM can use this before final confirmation, when the caller asks "what do I have so far?", or when it needs to read back the full order.

### Tool: `set_delivery_details`

```python
async def set_delivery_details(
    params: FunctionCallParams,
    recipient_name: str,
    address: str,
    delivery_date: str,
) -> None:
```

This tool stores delivery information in `order["delivery"]`.

The stored shape is:

```python
{
    "recipient_name": recipient_name,
    "address": address,
    "delivery_date": delivery_date,
}
```

The date is saved in the caller's own words. The code does not parse, validate, normalize, or calendar-check the date.

That choice keeps the starter simple. The system prompt includes today's date so the LLM can reason about relative phrases like "this Friday", but the backend tool still stores whatever final string the LLM passes.

For production, this is a likely upgrade point:

- Validate that the address is deliverable.
- Normalize dates to ISO format.
- Check delivery cutoff times.
- Ask for missing apartment/unit information.
- Add delivery fee calculations.

### Tool: `place_order`

```python
async def place_order(params: FunctionCallParams) -> None:
```

This tool finalizes the order. The docstring tells the LLM:

```text
Only call this after the customer has confirmed the items AND delivery details.
```

The implementation enforces two backend guards:

1. If there are no items:

   ```python
   {"ok": False, "reason": "No items in the order yet."}
   ```

2. If delivery details are missing:

   ```python
   {"ok": False, "reason": "Missing delivery details."}
   ```

If both checks pass:

1. Recompute the total.
2. Generate a fake confirmation number:

   ```python
   confirmation = f"FLW-{random.randint(100000, 999999)}"
   ```

3. Log the placed order.
4. Return:

   ```python
   {
       "ok": True,
       "confirmation_number": confirmation,
       "total": round(total, 2),
       "eta": "within 2 business days",
   }
   ```

No real order is sent anywhere. This is a mock backend completion point.

For a real shop, this function would call an order API, payment workflow, CRM, fulfillment system, or database transaction.

### Tool: `end_call`

```python
async def end_call(params: FunctionCallParams) -> None:
```

This tool ends the Pipecat task and therefore hangs up or disconnects the session.

Its docstring is intentionally strict:

```text
Only call this AFTER you have said goodbye to the customer in the same turn.
```

The implementation:

1. Logs that `end_call` was invoked.
2. Pushes an `EndTaskFrame` upstream:

   ```python
   await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
   ```

3. Returns `{"ok": True}` through `params.result_callback`.
4. Sets `FunctionCallResultProperties(run_llm=False)`.

The `run_llm=False` property is important. Normally, after a tool result comes back, the LLM may generate another response. Here, the model should already have spoken the goodbye in the same turn. Running the LLM again could create an awkward extra utterance during shutdown.

## Tool Registration

After defining the nested tools, `run_bot` collects them:

```python
tool_functions = [
    list_bouquets,
    check_availability,
    add_to_order,
    get_order_summary,
    set_delivery_details,
    place_order,
    end_call,
]
tools = ToolsSchema(standard_tools=tool_functions)
```

Later, each function is registered directly on the LLM:

```python
for fn in tool_functions:
    llm.register_direct_function(fn)
```

The tool function docstrings are not just comments. Pipecat uses them to describe the tools to the LLM, including when each tool should be called and what arguments it accepts. In this file, the docstrings are part of the bot's behavior.

## Caller Personalization

The bot uses `from_number` to decide whether the caller is known:

```python
customer = KNOWN_CUSTOMERS.get(from_number or "")
```

If the caller is known, the prompt includes:

- They are a returning customer.
- Their saved name.
- Their last order.
- A privacy-sensitive greeting rule.

The prompt explicitly says not to greet them by name or mention the last order immediately. Instead, it tells the assistant to greet generically and only offer the last order once the caller expresses intent to buy flowers.

This creates a more natural experience:

```text
Welcome back to Field & Flower! How can I help today?
```

Then later:

```text
I have you down for the rose romance last time, want that again or something different?
```

If the caller is not known, the prompt simply says they are a new customer and the bot should introduce the shop briefly.

## System Instruction

The `system_instruction` string is the bot's conversational contract. It contains business rules, style rules, tool-use rules, and call-ending rules.

Key behavior rules:

- The assistant is a Field & Flower order-taker.
- It must use tools for catalog lookup, stock checks, order mutation, delivery capture, and order placement.
- It must confirm the full order before calling `place_order`.

Voice style rules:

- Keep most turns to one or two short sentences.
- Ask one thing at a time.
- Avoid filler phrases like "Absolutely!" and "Perfect!"
- Describe bouquets plainly.
- Lead bouquet listings with the bouquet name.
- List no more than four or five options at once.
- Avoid unnecessary restatement.
- Use contractions and natural fragments.
- Do not use bullets or emojis because responses are spoken aloud.
- Read prices as words, not symbols.

Tool-use rules:

- If the caller mentions an occasion, pass it to `list_bouquets(occasion=...)`.
- If the caller asks about deals, call `list_bouquets(specials_only=True)`.
- Do not dump the whole catalog when a filtered subset is better.

Call-ending rule:

- When the order is done or the caller says goodbye, the assistant must say a short closing line and call `end_call` in the same turn.

Date awareness:

```python
f"Today is {date.today().strftime('%A, %B %d, %Y')}..."
```

This lets the LLM interpret relative dates in conversation. The code does not do date math itself.

## STT Service

The speech-to-text service is created like this:

```python
stt = NVidiaWebSocketSTTService(
    url=os.getenv("NVIDIA_ASR_URL", "ws://192.168.7.228:8081"),
    strip_interim_prefix=True,
)
```

This connects to a NVIDIA streaming ASR server over WebSocket.

Important assumptions:

- Input audio is 16-bit PCM.
- Default local/WebRTC input is 16 kHz mono.
- `NVIDIA_ASR_URL` can override the endpoint.
- `strip_interim_prefix=True` is enabled because this deployment emits cumulative interim transcripts. The STT service strips already-finalized tokens so the current turn is cleaner.

The STT service itself lives in `nvidia_stt.py`. `bot-nemotron.py` only instantiates it and places it in the pipeline.

## LLM Service

The LLM is configured like this:

```python
enable_thinking = os.getenv("NEMOTRON_ENABLE_THINKING", "false").lower() == "true"
llm = VLLMOpenAILLMService(
    api_key=os.getenv("NEMOTRON_LLM_API_KEY", "EMPTY"),
    base_url=os.getenv("NEMOTRON_LLM_URL", "http://192.168.7.228:8000/v1"),
    settings=VLLMOpenAILLMService.Settings(
        model=os.getenv("NEMOTRON_LLM_MODEL", "nvidia/nemotron-3-super"),
        system_instruction=system_instruction,
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}}},
    ),
)
```

The Nemotron model is served by vLLM behind an OpenAI-compatible `/v1` API. That is why the service subclasses Pipecat's `OpenAILLMService` rather than using an OpenAI Responses API service.

### Thinking mode

Nemotron can be run with reasoning/thinking enabled:

```bash
export NEMOTRON_ENABLE_THINKING=true
```

The bot passes that as:

```python
extra_body.chat_template_kwargs.enable_thinking
```

For voice, the file defaults this to false. The comments explain why: if the vLLM server does not separate reasoning into a non-spoken field such as `reasoning_content`, chain-of-thought could appear in the normal streamed `content` field and get spoken to the caller.

Keep thinking disabled unless the server is configured with an appropriate reasoning parser and you have verified that only final answer content reaches TTS.

### Why `VLLMOpenAILLMService` exists

`nemotron_llm.py` adjusts one metric: time to first byte/token for spoken output.

With reasoning models, the first streamed chunks may contain role metadata or reasoning, not user-visible answer text. Stock Pipecat may stop the TTFB clock too early. `VLLMOpenAILLMService` waits until the stream contains actual `content` or a tool call before allowing `stop_ttfb_metrics` to record.

This does not change the assistant's answers. It changes the latency metric so it reflects time to first spoken response.

## TTS Service

The bot uses Gradium TTS:

```python
tts = GradiumTTSService(
    api_key=os.environ["GRADIUM_API_KEY"],
    settings=GradiumTTSService.Settings(
        voice=os.getenv("GRADIUM_VOICE_ID", "Eu9iL_CYe8N-Gkx_"),
    ),
)
```

`GRADIUM_API_KEY` is required. The code uses `os.environ["GRADIUM_API_KEY"]`, so startup fails with `KeyError` if the key is missing.

`GRADIUM_VOICE_ID` is optional. If unset, the bot uses the hardcoded voice ID.

## Conversation Context and Turn Detection

The bot creates an LLM context with tools:

```python
context = LLMContext(tools=tools)
```

Then it creates user and assistant aggregators:

```python
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
        user_turn_strategies=FilterIncompleteUserTurnStrategies(),
    ),
)
```

The aggregators maintain chat history:

- The user aggregator turns transcriptions into user messages and decides when a user turn is complete.
- The assistant aggregator captures assistant output and tool results back into the context.

`SileroVADAnalyzer` detects speech activity. `FilterIncompleteUserTurnStrategies` helps avoid sending incomplete partial utterances to the LLM.

This is important for voice agents. Without careful turn handling, the model may respond before the caller finishes a sentence.

## Pipeline Assembly

The pipeline is:

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

Each component receives and emits Pipecat frames.

### Component by component

`transport.input()` receives audio frames from the browser or Twilio.

`stt` converts audio into interim and final transcription frames.

`user_aggregator` collects user text into a complete LLM turn and appends it to `context`.

`llm` sends the context to Nemotron. The LLM may return assistant text and/or tool calls.

`tts` converts assistant text into audio.

`transport.output()` sends generated audio back to the caller.

`assistant_aggregator` records assistant output in the shared context after it has passed through output.

The order matters. The assistant aggregator appears after output so spoken responses can be tracked as part of the conversation state without blocking the audio path unnecessarily.

## Pipeline Worker

The worker wraps the pipeline:

```python
worker = PipelineWorker(
    pipeline,
    params=PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
        audio_in_sample_rate=audio_in_sample_rate,
        audio_out_sample_rate=audio_out_sample_rate,
    ),
)
```

Metrics are enabled so Pipecat can report latency and usage data.

The sample rates come from `run_bot` arguments. Local WebRTC uses the defaults. Twilio passes overrides from `bot(...)`.

## Transport Event Handlers

Two event handlers are registered on the transport.

### `on_client_connected`

```python
@transport.event_handler("on_client_connected")
async def on_client_connected(transport, client):
```

When a client connects:

1. Log `"Client connected"`.
2. Add a synthetic user message to the context:

   ```python
   {
       "role": "user",
       "content": "A customer just called. Greet them, 'This is Field & Flower, your local flower shop. How can I help you today?'",
   }
   ```

3. Queue `LLMRunFrame()`:

   ```python
   await worker.queue_frames([LLMRunFrame()])
   ```

This starts the conversation without waiting for the caller to speak first. The model sees the synthetic instruction as a user message and generates the opening greeting.

### `on_client_disconnected`

```python
@transport.event_handler("on_client_disconnected")
async def on_client_disconnected(transport, client):
```

When the client disconnects:

1. Log `"Client disconnected"`.
2. Cancel the worker:

   ```python
   await worker.cancel()
   ```

Cancellation lets Pipecat stop the pipeline and clean up streaming resources.

## Running the Worker

At the end of `run_bot`:

```python
runner = WorkerRunner(handle_sigint=False)

await runner.add_workers(worker)
await runner.run()
```

`WorkerRunner` manages the pipeline worker lifecycle. `handle_sigint=False` means this runner does not install its own Ctrl-C signal handler; outer Pipecat runner code handles process-level behavior.

## Function: `bot(runner_args)`

```python
async def bot(runner_args: RunnerArguments):
```

This is the main Pipecat entrypoint. Pipecat's runner calls it with a `RunnerArguments` instance. The function inspects the argument type and creates the right transport.

It supports two transport modes:

- `SmallWebRTCRunnerArguments`: local browser testing.
- `WebSocketRunnerArguments`: Twilio media streams through a WebSocket.

If it receives any other type, it logs an error and returns.

### Shared setup

The function initializes:

```python
from_number: str | None = None
transport_overrides: dict = {}
```

`from_number` is filled only on the Twilio path, because local WebRTC does not have caller ID.

`transport_overrides` is used to pass Twilio-specific sample rates into `run_bot`.

### Krisp filter

```python
if os.environ.get("ENV") != "local":
    from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
    krisp_filter = KrispVivaFilter()
else:
    krisp_filter = None
```

Krisp noise filtering is enabled outside local development. When `ENV=local`, the filter is disabled.

The import is inside the conditional because Krisp is expected in the Pipecat Cloud deployment environment, not necessarily in every local setup.

## WebRTC Transport Path

This branch handles local browser use:

```python
case SmallWebRTCRunnerArguments():
```

The function extracts the WebRTC connection:

```python
webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
```

Then creates a `SmallWebRTCTransport`:

```python
transport = SmallWebRTCTransport(
    webrtc_connection=webrtc_connection,
    params=TransportParams(
        audio_in_enabled=True,
        audio_in_filter=krisp_filter,
        audio_out_enabled=True,
    ),
)
```

This transport:

- Receives microphone audio from the browser.
- Optionally applies the Krisp audio filter.
- Sends synthesized audio back to the browser.

No caller lookup is performed in this mode.

Finally, `bot` calls:

```python
await run_bot(transport, from_number=from_number, **transport_overrides)
```

Because `from_number` is still `None` and `transport_overrides` is empty, `run_bot` uses new-customer prompting and WebRTC sample-rate defaults.

## Twilio WebSocket Transport Path

This branch handles production phone calls:

```python
case WebSocketRunnerArguments():
```

### Sample rate overrides

Twilio media streams use 8 kHz mu-law audio, so the bot overrides the pipeline rates:

```python
transport_overrides["audio_in_sample_rate"] = 8000
transport_overrides["audio_out_sample_rate"] = 8000
```

These values are passed into `run_bot`.

### Parse telephony metadata

```python
_, call_data = await parse_telephony_websocket(runner_args.websocket)
```

`call_data` contains identifiers such as:

- `call_id`
- `stream_id`

The exact values come from Twilio's media stream connection.

### Fetch caller info

```python
call_info = await get_call_info(call_data["call_id"])
```

If this succeeds, the bot extracts:

```python
from_number = call_info.get("from_number")
```

Then it logs the source and destination number. That `from_number` later drives the known-customer prompt.

### Twilio serializer

```python
serializer = TwilioFrameSerializer(
    stream_sid=call_data["stream_id"],
    call_sid=call_data["call_id"],
    account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
    auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
)
```

The serializer converts between Pipecat audio frames and Twilio media stream messages.

It needs:

- The Twilio stream SID.
- The Twilio call SID.
- Twilio credentials.

### FastAPI WebSocket transport

```python
transport = FastAPIWebsocketTransport(
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

This transport:

- Reads audio from the Twilio WebSocket.
- Applies the optional Krisp filter.
- Writes TTS audio back through the same WebSocket.
- Uses the Twilio serializer.
- Does not add WAV headers, because Twilio expects raw stream payloads.

After this branch, `bot` calls `run_bot` with the Twilio transport, caller ID if available, and the 8 kHz audio overrides.

## Entrypoint: `if __name__ == "__main__"`

The file ends with:

```python
if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
```

When you run:

```bash
uv run bot-nemotron.py
```

Pipecat's `main()` discovers and runs the module-level `bot(runner_args)` function. The runner handles the surrounding server setup, including the local WebRTC page at `localhost:7860`.

## End-to-End Conversation Flow

A typical local browser conversation looks like this:

1. User runs `uv run bot-nemotron.py`.
2. Pipecat starts the local runner.
3. User opens `http://localhost:7860`.
4. User clicks Connect.
5. `bot(...)` receives `SmallWebRTCRunnerArguments`.
6. `bot(...)` creates a `SmallWebRTCTransport`.
7. `bot(...)` calls `run_bot(...)`.
8. `run_bot(...)` creates per-call order state.
9. `run_bot(...)` defines and registers tools.
10. `run_bot(...)` creates the system prompt.
11. `run_bot(...)` creates STT, LLM, and TTS services.
12. `run_bot(...)` assembles and starts the pipeline worker.
13. `on_client_connected` adds a synthetic greeting request and queues `LLMRunFrame`.
14. Nemotron generates the greeting.
15. Gradium converts it to audio.
16. The transport plays the greeting to the caller.
17. The caller asks for flowers.
18. STT transcribes the caller.
19. The user aggregator sends the completed user turn to Nemotron.
20. Nemotron either responds directly or calls a tool such as `list_bouquets`.
21. Tool results are returned to Nemotron.
22. Nemotron speaks a concise answer.
23. The loop continues until `place_order` and eventually `end_call`.

## Customizing the Bot

### Change the catalog

Edit `BOUQUETS` in `mock_backend.py`.

Each key should be lowercase because tool calls normalize bouquet names with `.lower()`:

```python
"birthday brights": {
    "price": 48.00,
    "description": "Sunflowers, gerbera daisies, and orange roses",
    "in_stock": True,
    "occasions": ["birthday", "congratulations", "thank you"],
    "on_special": True,
}
```

If you add new occasion categories, update the `list_bouquets` docstring and system prompt examples so the LLM knows to use them.

### Change known customers

Edit `KNOWN_CUSTOMERS` in `mock_backend.py`:

```python
KNOWN_CUSTOMERS = {
    "+14155551234": {"name": "Alex", "last_order": "rose romance"},
}
```

Use E.164 phone number format because Twilio's `from` field usually arrives that way.

### Add a new tool

To add a new tool:

1. Define an async function inside `run_bot`.
2. Make `params: FunctionCallParams` the first argument.
3. Give it a clear docstring that tells the LLM when to call it.
4. Return structured output through `await params.result_callback(...)`.
5. Add the function to `tool_functions`.

Example shape:

```python
async def apply_coupon(params: FunctionCallParams, code: str) -> None:
    """Apply a coupon code to the current order.

    Only call this when the customer provides a coupon or asks to use a discount.

    Args:
        code: Coupon code exactly as spoken by the customer.
    """
    ...
    await params.result_callback({"ok": True, "discount": 10.0})
```

Because `ToolsSchema` and `register_direct_function` iterate over `tool_functions`, adding it to that list exposes it to the model and wires the handler.

### Replace the mock backend

The nested tools are the boundary between the LLM and backend systems. To use a real backend, replace the internals of functions like:

- `list_bouquets`
- `check_availability`
- `add_to_order`
- `set_delivery_details`
- `place_order`

Keep the tool outputs simple and explicit. LLMs handle compact JSON-like result structures better than verbose backend payloads.

### Change the personality

Edit `system_instruction`. Keep voice constraints explicit. Voice agents need different prompting than chatbots:

- Short responses.
- One question per turn.
- No visual formatting.
- Clear tool-use policies.
- Clear hang-up policy.

## Common Failure Modes

### The bot starts but never speaks

Check:

- `GRADIUM_API_KEY` is set.
- The browser or Twilio transport connected.
- `on_client_connected` fired in logs.
- The LLM endpoint in `NEMOTRON_LLM_URL` is reachable.

### The bot cannot transcribe the caller

Check:

- `NVIDIA_ASR_URL` points to a running WebSocket STT service.
- The input sample rate is correct for the transport.
- Browser microphone permissions are granted.
- Twilio audio is being serialized through `TwilioFrameSerializer`.

### The assistant says internal reasoning out loud

Set:

```bash
export NEMOTRON_ENABLE_THINKING=false
```

Only enable thinking after verifying the vLLM server separates reasoning from normal spoken `content`.

### Known-customer personalization does not work

Check:

- You are using the Twilio path, not local WebRTC.
- `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` are set.
- `get_call_info` returns a `from_number`.
- The number exists in `KNOWN_CUSTOMERS` in E.164 format.

### The bot lists too many bouquets

Improve either:

- Occasion tags in `mock_backend.py`.
- The `list_bouquets` docstring.
- The system instruction that tells the LLM to filter by occasion or specials.

The model decides when to call tools, so tool descriptions and prompt rules directly affect behavior.

## Mental Model

The most important thing to understand is that `bot-nemotron.py` is not a traditional request/response web handler. It is a streaming pipeline:

- Audio flows in continuously.
- STT emits transcription frames.
- Aggregators decide when a user turn is ready.
- The LLM receives accumulated context plus tool schemas.
- Tool calls execute local Python functions.
- Assistant text streams into TTS.
- TTS audio streams back to the caller.
- Event handlers start and stop the lifecycle.

The nested tool functions are the application domain. The Pipecat pipeline is the real-time voice infrastructure. The system prompt is the operating policy that tells the LLM how to use both.
