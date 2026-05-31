# Bright Smile Dental Voice Agent

**Voice DEMO Video Link**: https://youtu.be/SfJ8fjiw8K0

Bright Smile Dental Voice Agent is a front-desk voice assistant for a dental
office. The agent, Aria, answers routine calls over browser WebRTC or Twilio,
helps callers book and reschedule appointments, checks a small accepted-insurance
list, and routes clinical or emergency questions safely.

This repository has been evolving from a hackathon starter into a more reliable
and faster voice-agent system. The numbered bots in `server/` are not separate
products. They are a deliberate optimization sequence:

- `bot0.py`: prove the full voice stack works end to end.
- `bot1.py`: make caller-perceived latency measurable.
- `bot2.py`: tune the runtime path and add a narrow deterministic booking path.
- `bot3.py`: move common front-desk workflows into short-lived call memory, with
  the LLM as fallback.

The strategic goal is not just to make the model answer better. It is to make the
whole call experience dependable: faster first responses, fewer missed user
turns, safer business-rule handling, and clearer evidence about why a call passed
or failed.

## Executive Summary

We have been optimizing Bright Smile around one core product question:

> Can a caller complete safe front-desk dental work quickly, without the agent
> inventing authority, losing context, or making the caller wait too long?

The work across `bot0.py`, `bot1.py`, `bot2.py`, and `bot3.py` has focused on
five reliability and speed levers:

1. **End-to-end baseline reliability.** `bot0.py` establishes the production
   shape: speech-to-text, turn aggregation, Nemotron LLM, Gradium text-to-speech,
   tool-backed appointment actions, browser calls, and Twilio calls.

2. **Measurement before rewriting.** `bot1.py` adds voice latency logging so the
   team can distinguish "the agent feels slow" from specific problems such as
   VAD delay, transcript finalization, LLM response time, tool latency, or TTS
   startup.

3. **Lower latency for predictable work.** `bot2.py` adds stage visibility and a
   narrow appointment fast path. Simple bookings should not always wait for a
   full LLM tool-use cycle when the requested workflow is structured and safe.

4. **Stateful front-desk behavior.** `bot3.py` adds short-lived call memory so
   Aria can remember what the caller already said, handle corrections, offer
   appointment slots, reschedule safely, and answer common policy questions
   consistently.

5. **Evaluation-driven tradeoffs.** The batch evaluation harness compares bot
   versions by pass rate, failing scenario IDs, judge reasons, infrastructure
   failures, and voice latency. A bot is better only if it improves task success
   without making the live call feel worse.

The direction is clear: keep the LLM for language flexibility, but move stable
business workflows into observable tools, deterministic processors, and per-call
memory. This reduces latency, reduces hallucination risk, and makes failures
debuggable.

## Product Goal

Bright Smile needs a voice agent that behaves like a competent front desk, not a
general medical assistant. Aria should:

- Greet callers naturally.
- Book appointments after collecting the required details.
- Reschedule appointments only when the caller has a valid confirmation ID.
- Check known accepted insurance providers.
- Decline unsupported actions such as cancellation.
- Avoid medical, dental, diagnosis, treatment, and medication advice.
- Route urgent symptoms toward emergency care first.
- Keep turns short and spoken, with one question at a time.

The agent is intentionally bounded. Reliability comes from knowing what Aria can
do, what must be tool-backed, and what must be refused or escalated.

## Reliability Strategy

### Tools Own Business State

The model can talk, but bookings and reschedules must go through backend tools.
That keeps the agent from saying "you are booked" unless the system has actually
created a confirmation.

The same principle applies to insurance. The accepted list is intentionally
small. Unknown providers should be handled as "the office can confirm," not as a
guessed yes.

### Voice Quality Is A Runtime Problem

A voice agent fails differently from a text chatbot. It can be wrong because the
model misunderstood the task, but it can also be wrong because:

- VAD waited too long or stopped too early.
- STT finalized the wrong transcript.
- The caller interrupted the agent.
- The LLM took too long to produce a usable answer.
- TTS startup made the response feel delayed.

That is why the later bots add latency and stage logging. The team needs to know
where time and uncertainty enter the call path.

### Deterministic Paths Should Stay Narrow

Common front-desk workflows are predictable enough to optimize. Appointment
booking, slot selection, rescheduling with a confirmation ID, insurance checks,
and safe refusal language do not need unlimited model freedom every turn.

The rule is: deterministic logic should handle what it knows, use the same tools
as the LLM path, and hand back to the LLM when the caller says something outside
the tested workflow.

### Memory Is Per Call, Not Identity

`bot3.py` remembers details inside a single conversation: name, reason, date,
time, confirmation ID, offered slots, and the last question asked. That makes the
agent feel more competent because it stops asking for information the caller
already gave.

This is not patient identity. Caller ID is for logging context only. Aria should
not claim to recognize a patient or pull up records from phone number alone.

## Bot Evolution

### `bot0.py`: Working Baseline

`bot0.py` proves the complete live voice-agent path. It connects browser and
Twilio transports to NVIDIA streaming STT, Nemotron, Gradium TTS, and the Bright
Smile tool set.

Its value is the baseline. Its limitation is that when the call feels slow or
unreliable, it does not give enough evidence about where the problem happened.

### `bot1.py`: Measurable Latency

`bot1.py` keeps the product behavior nearly the same and adds latency logging.
The important metrics are:

- **TTFA:** time to first audio after the caller stops speaking.
- **TTLA:** time to last audio after the caller stops speaking.

This turns voice quality from a subjective complaint into something the team can
compare across bot versions and scenarios.

### `bot2.py`: Runtime Tuning And Fast Path

`bot2.py` adds stage-level visibility and makes voice runtime settings easier to
tune by channel. It also introduces a narrow appointment fast path for simple,
structured booking turns.

The product bet is that predictable work should be faster and less fragile than
waiting for full LLM reasoning every time. The risk is overreach, so the fast
path stays focused and still uses the same booking tools.

### `bot3.py`: Memory-First Front Desk

`bot3.py` expands the deterministic layer into a memory-first front-desk
processor. It handles common workflows before the LLM:

- Booking with partial information.
- Caller corrections.
- Vague time requests and slot options.
- Rescheduling with confirmation IDs.
- Known insurance checks.
- Emergency and medical-advice guardrails.
- Unsupported cancellation requests.

The LLM remains available for open-ended language. The improvement is that Aria
does not need to rediscover basic front-desk state on every turn.

## What We Optimize For

The target is not the shortest possible response at any cost. The target is a
better front-desk call. The core scorecard is:

- Higher scenario pass rate.
- Fewer high-severity failures.
- Lower TTFA and TTLA, especially p95.
- Fewer infrastructure failures.
- Correct tool-backed confirmations.
- Safe refusals for clinical, cancellation, and privacy boundaries.
- Better handling of corrections, partial answers, and follow-up questions.

A fast bot that fails to book correctly is not a win. A correct bot that makes
callers wait too long is also not good enough. The work is to move both curves:
more reliable task completion and lower caller-perceived delay.

## Evaluation

The evaluation harness in `server/` runs scripted caller scenarios against one or
more bot versions, then records transcripts, judge reasons, tool calls, pass
rates, and latency metrics.

Run a small comparison:

```bash
cd server
uv run python batch_eval_runner.py --bots bot0.py bot1.py bot2.py bot3.py --limit 3 --max-workers 1
```

Run a targeted scenario:

```bash
cd server
uv run python batch_eval_runner.py --bots bot2.py bot3.py --scenario medical_advice_ibuprofen --max-workers 1
```

The most useful review loop is:

1. Compare pass rate and p95 latency.
2. Read failing scenario IDs and judge reasons.
3. Inspect transcripts and tool calls.
4. Use stage latency when the failure looks timing-related.
5. Fix the narrowest layer that owns the failure: prompt, tool, fast path,
   memory, VAD, transport, or eval.

## Local Development

The active Python project lives in `server/`.

```bash
cd server
uv sync
```

Configure service keys in `server/.env`. At minimum, local voice runs need the
STT, LLM, and TTS service configuration used by the selected bot.

Run a bot locally:

```bash
cd server
uv run bot3.py
```

Then open the local WebRTC page shown by the Pipecat runner and connect from the
browser. For phone testing, use the Twilio websocket deployment path configured
for Pipecat Cloud.

Run tests:

```bash
cd server
uv run pytest
```

## Documentation Map

For deeper implementation notes, see:

- `server/docs/README.md` for the detailed bot evolution narrative.
- `server/docs/EVALUATION_TUTORIAL.md` for evaluation workflow details.
- `server/docs/BATCH_EVALUATION_TUTORIAL.md` for batch comparisons.
- `server/notes/latency.md` for latency-analysis notes.

The root README is intentionally strategic. The detailed docs are useful when
changing code. This file should stay focused on what we are optimizing and why.
