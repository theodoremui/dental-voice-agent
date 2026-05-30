from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DEFAULT_HOST = "127.0.0.1"
DEFAULT_START_PORT = 7860
DEFAULT_ARTIFACTS_DIR = "batch_eval_artifacts"
DEFAULT_OUTPUT_PATH = "batch_results.json"
DEFAULT_RUNS_PATH = "batch_runs.jsonl"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 90.0
PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BotRunSpec:
    index: int
    label: str
    bot_path: Path
    host: str
    port: int
    bot_url: str
    work_dir: Path
    result_path: Path
    eval_runs_path: Path
    latency_path: Path
    bot_log_path: Path
    eval_log_path: Path

    def metadata(self) -> dict[str, Any]:
        return {
            "bot": self.label,
            "bot_index": self.index,
            "bot_path": str(self.bot_path),
            "bot_port": self.port,
            "bot_url": self.bot_url,
        }


@dataclass
class BotEvalOutcome:
    spec: BotRunSpec
    summary: dict[str, Any]
    result: dict[str, Any] | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run eval_runner.py across multiple voice-agent bot implementations in "
            "parallel. Unknown CLI options are forwarded to eval_runner.py."
        )
    )
    parser.add_argument(
        "--bots",
        nargs="+",
        required=True,
        help="Bot scripts to evaluate, for example: --bots bot0.py bot1.py bot2.py",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional plot-friendly labels matching --bots order. Defaults to file stems.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Maximum bots to evaluate concurrently. Default: all bots.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host used for local Pipecat bot runners. Default: {DEFAULT_HOST}.",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=int(os.getenv("BATCH_EVAL_START_PORT", str(DEFAULT_START_PORT))),
        help=f"First port to try when --ports is omitted. Default: {DEFAULT_START_PORT}.",
    )
    parser.add_argument(
        "--ports",
        nargs="+",
        type=int,
        help="Explicit per-bot ports. Count must match --bots.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=float(
            os.getenv("BATCH_EVAL_STARTUP_TIMEOUT_SECONDS", DEFAULT_STARTUP_TIMEOUT_SECONDS)
        ),
        help=f"Seconds to wait for each bot /status endpoint. Default: {DEFAULT_STARTUP_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=0.0,
        help="Optional seconds to allow each eval_runner.py process. Default: no timeout.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory for per-bot results, latency logs, and process logs. Default: {DEFAULT_ARTIFACTS_DIR}.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Aggregate batch JSON path. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--runs",
        default=DEFAULT_RUNS_PATH,
        help=f"Trend JSONL path. One row is appended per bot. Default: {DEFAULT_RUNS_PATH}.",
    )
    parser.add_argument(
        "--eval-runner",
        default="eval_runner.py",
        help="Path to the single-bot eval runner. Default: eval_runner.py.",
    )
    parser.add_argument(
        "--uv",
        default="uv",
        help="uv executable to use when launching bots and eval_runner.py. Default: uv.",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = _build_parser()
    args, eval_args = parser.parse_known_args(argv)
    if eval_args[:1] == ["--"]:
        eval_args = eval_args[1:]
    return args, eval_args


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._").lower()
    return slug or "bot"


def _unique_labels(bot_paths: list[Path], labels: list[str] | None) -> list[str]:
    if labels is not None and len(labels) != len(bot_paths):
        raise SystemExit("--labels count must match --bots count.")

    base_labels = labels or [path.stem for path in bot_paths]
    counts: dict[str, int] = {}
    unique = []
    for label in base_labels:
        slug = _slug(label)
        counts[slug] = counts.get(slug, 0) + 1
        unique.append(slug if counts[slug] == 1 else f"{slug}-{counts[slug]}")
    return unique


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _allocate_ports(host: str, count: int, start_port: int) -> list[int]:
    if not 0 < start_port < 65536:
        raise SystemExit("--start-port must be between 1 and 65535.")

    ports: list[int] = []
    candidate = start_port
    while len(ports) < count:
        if candidate > 65535:
            raise SystemExit(f"Could not allocate {count} free ports starting at {start_port}.")
        if candidate not in ports and _port_is_free(host, candidate):
            ports.append(candidate)
        candidate += 1
    return ports


def _build_specs(args: argparse.Namespace, *, batch_id: str) -> list[BotRunSpec]:
    bot_paths = [Path(raw).expanduser().resolve() for raw in args.bots]
    missing = [str(path) for path in bot_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Bot file(s) not found: {', '.join(missing)}")

    if args.ports is not None:
        if len(args.ports) != len(bot_paths):
            raise SystemExit("--ports count must match --bots count.")
        if len(set(args.ports)) != len(args.ports):
            raise SystemExit("--ports values must be unique.")
        invalid_ports = [port for port in args.ports if not 0 < port < 65536]
        if invalid_ports:
            raise SystemExit(f"--ports values must be between 1 and 65535: {invalid_ports}")
        occupied_ports = [port for port in args.ports if not _port_is_free(args.host, port)]
        if occupied_ports:
            raise SystemExit(f"--ports values already in use: {occupied_ports}")
        ports = args.ports
    else:
        ports = _allocate_ports(args.host, len(bot_paths), args.start_port)

    labels = _unique_labels(bot_paths, args.labels)
    batch_dir = Path(args.artifacts_dir).expanduser().resolve() / batch_id
    specs = []
    for index, (bot_path, label, port) in enumerate(zip(bot_paths, labels, ports), start=1):
        work_dir = batch_dir / f"{index:02d}-{label}"
        specs.append(
            BotRunSpec(
                index=index,
                label=label,
                bot_path=bot_path,
                host=args.host,
                port=port,
                bot_url=f"http://{args.host}:{port}",
                work_dir=work_dir,
                result_path=work_dir / "results.json",
                eval_runs_path=work_dir / "runs.jsonl",
                latency_path=work_dir / "latency.jsonl",
                bot_log_path=work_dir / "bot.log",
                eval_log_path=work_dir / "eval.log",
            )
        )
    return specs


def _read_status(bot_url: str) -> dict[str, Any]:
    with urlopen(f"{bot_url}/status", timeout=2.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


async def _wait_for_ready(
    spec: BotRunSpec,
    process: asyncio.subprocess.Process,
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.returncode is not None:
            raise RuntimeError(
                f"{spec.label} exited before becoming ready with code {process.returncode}. "
                f"See {spec.bot_log_path}."
            )
        try:
            status = await asyncio.to_thread(_read_status, spec.bot_url)
        except (HTTPError, OSError, URLError, TimeoutError) as exc:
            last_error = exc
        else:
            transports = status.get("transports", [])
            if status.get("status") == "ready" and "webrtc" in transports:
                return
            last_error = RuntimeError(f"unexpected /status payload: {status!r}")
        await asyncio.sleep(0.5)

    detail = f" Last error: {last_error}" if last_error else ""
    raise TimeoutError(
        f"{spec.label} did not become ready at {spec.bot_url} within "
        f"{timeout_seconds:.1f}s.{detail} See {spec.bot_log_path}."
    )


def _bot_command(args: argparse.Namespace, spec: BotRunSpec) -> list[str]:
    return [
        args.uv,
        "run",
        str(spec.bot_path),
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "-t",
        "webrtc",
    ]


def _eval_command(
    args: argparse.Namespace,
    spec: BotRunSpec,
    eval_args: list[str],
) -> list[str]:
    return [
        args.uv,
        "run",
        "python",
        args.eval_runner,
        *eval_args,
        "--bot-url",
        spec.bot_url,
        "--output",
        str(spec.result_path),
        "--runs",
        str(spec.eval_runs_path),
        "--latency-path",
        str(spec.latency_path),
    ]


async def _terminate_process(process: asyncio.subprocess.Process, *, label: str) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=10.0)
    except TimeoutError:
        process.kill()
        await process.wait()
        print(f"{label}: killed after graceful termination timed out", file=sys.stderr)


async def _run_logged_process(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: float,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_file:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=PROJECT_DIR,
            env=env,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            if timeout_seconds > 0:
                return await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            return await process.wait()
        except TimeoutError:
            await _terminate_process(process, label=command[0])
            raise


def _load_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _tail(path: Path, *, lines: int = 30) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def _inject_agent_metadata(result: dict[str, Any], spec: BotRunSpec) -> dict[str, Any]:
    enriched = dict(result)
    metadata = spec.metadata()
    enriched["evaluated_agent"] = metadata
    enriched["agent_source"] = f"{metadata['bot']} ({metadata['bot_path']}) at {metadata['bot_url']}"
    enriched["bot"] = metadata["bot"]
    enriched["bot_path"] = metadata["bot_path"]
    return enriched


def _summary_from_result(
    spec: BotRunSpec,
    result: dict[str, Any] | None,
    *,
    batch_id: str,
    batch_timestamp: float,
    status: str,
    duration_ms: int,
    error: str | None = None,
) -> dict[str, Any]:
    live_latency = result.get("live_latency", {}) if result else {}
    scenarios = result.get("scenarios", []) if result else []
    failure_ids = [
        str(scenario.get("id"))
        for scenario in scenarios
        if isinstance(scenario, dict) and not scenario.get("passed")
    ]
    summary = {
        "timestamp": result.get("timestamp", batch_timestamp) if result else batch_timestamp,
        "batch_id": batch_id,
        **spec.metadata(),
        "status": status,
        "run_id": result.get("run_id") if result else None,
        "mode": result.get("mode") if result else None,
        "scenario_count": result.get("scenario_count") if result else None,
        "passed": result.get("passed") if result else None,
        "pass_rate": result.get("pass_rate") if result else None,
        "p95_agent_reply_ms": result.get("p95_agent_reply_ms") if result else None,
        "p95_latency_ms": result.get("p95_latency_ms") if result else None,
        "live_p95_ttfa_ms": live_latency.get("p95_ttfa_ms"),
        "live_p95_ttla_ms": live_latency.get("p95_ttla_ms"),
        "live_completed_turns": live_latency.get("completed_turns"),
        "failure_count": len(failure_ids),
        "failure_ids": failure_ids,
        "duration_ms": duration_ms,
        "result_path": str(spec.result_path),
        "latency_path": str(spec.latency_path),
        "bot_log_path": str(spec.bot_log_path),
        "eval_log_path": str(spec.eval_log_path),
    }
    if error:
        summary["error"] = error
    return summary


async def _run_one_bot_eval(
    args: argparse.Namespace,
    eval_args: list[str],
    spec: BotRunSpec,
    *,
    batch_id: str,
    batch_timestamp: float,
) -> BotEvalOutcome:
    started = time.perf_counter()
    spec.work_dir.mkdir(parents=True, exist_ok=True)
    bot_env = os.environ.copy()
    bot_env["BOT_EVAL_SERVER"] = "0"
    bot_env["LATENCY_METRICS_PATH"] = str(spec.latency_path)

    bot_log_file = spec.bot_log_path.open("wb")
    bot_process: asyncio.subprocess.Process | None = None
    try:
        bot_process = await asyncio.create_subprocess_exec(
            *_bot_command(args, spec),
            cwd=PROJECT_DIR,
            env=bot_env,
            stdout=bot_log_file,
            stderr=asyncio.subprocess.STDOUT,
        )
        await _wait_for_ready(spec, bot_process, timeout_seconds=args.startup_timeout)

        eval_env = os.environ.copy()
        eval_env["EVAL_BOT_URL"] = spec.bot_url
        eval_env["LATENCY_METRICS_PATH"] = str(spec.latency_path)
        eval_code = await _run_logged_process(
            _eval_command(args, spec, eval_args),
            env=eval_env,
            log_path=spec.eval_log_path,
            timeout_seconds=args.eval_timeout,
        )
        result = _load_result(spec.result_path)
        if result is not None:
            result = _inject_agent_metadata(result, spec)
            _write_json(spec.result_path, result)

        duration_ms = int(round((time.perf_counter() - started) * 1000.0))
        if eval_code != 0:
            error = f"eval_runner.py exited with code {eval_code}. See {spec.eval_log_path}."
            if tail := _tail(spec.eval_log_path):
                error = f"{error}\n{tail}"
            return BotEvalOutcome(
                spec=spec,
                result=result,
                summary=_summary_from_result(
                    spec,
                    result,
                    batch_id=batch_id,
                    batch_timestamp=batch_timestamp,
                    status="error",
                    duration_ms=duration_ms,
                    error=error,
                ),
            )

        return BotEvalOutcome(
            spec=spec,
            result=result,
            summary=_summary_from_result(
                spec,
                result,
                batch_id=batch_id,
                batch_timestamp=batch_timestamp,
                status="completed",
                duration_ms=duration_ms,
            ),
        )
    except Exception as exc:
        duration_ms = int(round((time.perf_counter() - started) * 1000.0))
        return BotEvalOutcome(
            spec=spec,
            result=None,
            summary=_summary_from_result(
                spec,
                None,
                batch_id=batch_id,
                batch_timestamp=batch_timestamp,
                status="error",
                duration_ms=duration_ms,
                error=str(exc),
            ),
        )
    finally:
        if bot_process is not None:
            await _terminate_process(bot_process, label=spec.label)
        bot_log_file.close()


async def _run_batch(
    args: argparse.Namespace,
    eval_args: list[str],
    specs: list[BotRunSpec],
    *,
    batch_id: str,
    batch_timestamp: float,
) -> list[BotEvalOutcome]:
    jobs = len(specs) if args.jobs <= 0 else min(args.jobs, len(specs))
    semaphore = asyncio.Semaphore(jobs)

    async def run_limited(spec: BotRunSpec) -> BotEvalOutcome:
        async with semaphore:
            print(f"[{spec.index}/{len(specs)}] starting {spec.label} on {spec.bot_url}")
            outcome = await _run_one_bot_eval(
                args,
                eval_args,
                spec,
                batch_id=batch_id,
                batch_timestamp=batch_timestamp,
            )
            rate = outcome.summary.get("pass_rate")
            status = outcome.summary["status"]
            print(f"[{spec.index}/{len(specs)}] {spec.label}: {status} pass_rate={rate}")
            return outcome

    tasks = [asyncio.create_task(run_limited(spec)) for spec in specs]
    outcomes = [await task for task in asyncio.as_completed(tasks)]
    return sorted(outcomes, key=lambda outcome: outcome.spec.index)


def _build_batch_output(
    outcomes: list[BotEvalOutcome],
    *,
    batch_id: str,
    batch_timestamp: float,
    duration_ms: int,
    jobs: int,
    eval_args: list[str],
) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "timestamp": batch_timestamp,
        "duration_ms": duration_ms,
        "bot_count": len(outcomes),
        "completed": sum(1 for outcome in outcomes if outcome.summary["status"] == "completed"),
        "errors": sum(1 for outcome in outcomes if outcome.summary["status"] == "error"),
        "jobs": jobs,
        "eval_args": eval_args,
        "bots": [
            {
                "summary": outcome.summary,
                "result": outcome.result,
            }
            for outcome in outcomes
        ],
    }


def main(argv: list[str] | None = None) -> None:
    args, eval_args = _parse_args(argv)
    batch_id = time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:6]}"
    batch_timestamp = time.time()
    specs = _build_specs(args, batch_id=batch_id)
    jobs = len(specs) if args.jobs <= 0 else min(args.jobs, len(specs))

    print(
        f"Running batch {batch_id}: {len(specs)} bot(s), jobs={jobs}, "
        f"eval args={' '.join(eval_args) or '(none)'}"
    )

    started = time.perf_counter()
    outcomes = asyncio.run(
        _run_batch(
            args,
            eval_args,
            specs,
            batch_id=batch_id,
            batch_timestamp=batch_timestamp,
        )
    )
    duration_ms = int(round((time.perf_counter() - started) * 1000.0))
    output = _build_batch_output(
        outcomes,
        batch_id=batch_id,
        batch_timestamp=batch_timestamp,
        duration_ms=duration_ms,
        jobs=jobs,
        eval_args=eval_args,
    )

    output_path = Path(args.output).expanduser()
    runs_path = Path(args.runs).expanduser()
    _write_json(output_path, output)
    for outcome in outcomes:
        _append_jsonl(runs_path, outcome.summary)

    print(f"Wrote aggregate results to {output_path}")
    print(f"Appended trend rows to {runs_path}")

    if output["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
