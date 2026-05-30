from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{value}ms"


def _fmt_ts(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "unknown-time"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval results and trend rows.")
    parser.add_argument("--results", default="results.json", help="Path to results.json.")
    parser.add_argument("--runs", default="runs.jsonl", help="Path to runs.jsonl.")
    parser.add_argument("--limit", type=int, default=10, help="Trend rows to show.")
    args = parser.parse_args()

    results = _load_json(Path(args.results))
    if not results:
        raise SystemExit(f"No results found at {args.results}. Run eval_runner.py first.")

    print("Latest Results")
    print("--------------")
    print(f"run_id: {results.get('run_id')}")
    print(f"timestamp: {_fmt_ts(results.get('timestamp'))}")
    print(f"pass_rate: {results.get('pass_rate')}")
    print(f"passed: {results.get('passed')}/{results.get('scenario_count')}")
    print(f"text p95 agent reply: {_fmt_ms(results.get('p95_agent_reply_ms'))}")
    print(f"live p95 latency: {_fmt_ms(results.get('p95_latency_ms'))}")

    failures = [scenario for scenario in results.get("scenarios", []) if not scenario.get("passed")]
    if failures:
        print("\nFailures")
        print("--------")
        for scenario in failures:
            print(f"- {scenario.get('id')}: {scenario.get('reason')}")

    trend = _load_jsonl(Path(args.runs), max(1, args.limit))
    if trend:
        print("\nTrend")
        print("-----")
        for row in trend:
            print(
                f"{_fmt_ts(row.get('timestamp'))}  "
                f"pass_rate={row.get('pass_rate')}  "
                f"text_p95={_fmt_ms(row.get('p95_agent_reply_ms'))}  "
                f"live_p95={_fmt_ms(row.get('p95_latency_ms'))}"
            )


if __name__ == "__main__":
    main()
