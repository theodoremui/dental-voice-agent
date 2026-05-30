Ultraplan to modify the "eval_runner.py" to use the "bot.py" voice agent orchestrated by pipecat (cloud).  

Goal: simulated callers drive the same prompt + tools at text speed, a judge scores each against criteria, and you write results.json. Text-speed lets you run 25 scenarios in seconds and iterate all afternoon.

Pick scenario · run all ~25 Caller sim turn LLM playing the persona Agent reply same SYSTEM_PROMPT + TOOLS (tools.py) repeat each turn Judge criteria met? ✓ pass ✗ fail results.json pass-rate · p95 · per-scenario reasons dashboard.py pass-rate · p95 · trend iterate edit prompt 55% → 90%+ What text-level evals do and don't cover. They rigorously test reasoning, tool use, and guardrails — most of "production-grade correctness." They do not test ASR accuracy or audio latency; that's what latency.jsonl (live calls) and any voice-level Cekura runs are for. Say this out loud in the demo — judges respect knowing the boundary.

Create an "EVALUATION_TUTORIAL.md" with step-by-step clear and simple instructions on how to run the eval_runner.py and what to look for in the output.
