import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from scenarios import SCENARIOS

DEFAULT_ASR_URL = "ws://192.168.7.228:8081"
DEFAULT_TTS_ACTIVE_SESSION_LIMIT = 3
MAX_TURNS = 8
REPO_ROOT = Path(__file__).resolve().parent
JUDGE_POLICY_VERSION = "answer-present-v1"
JUDGE_POLICY_SUMMARY = (
    "Lenient answer-present policy: pass when the correct answer, action, or safe refusal "
    "appears anywhere in the observable agent response."
)


def load_batch_environment(env_root: Path = REPO_ROOT) -> None:
    load_dotenv(env_root / ".env", override=True)


@dataclass(frozen=True)
class BotSpec:
    id: str
    name: str
    passed_path: str
    resolved_path: Path
    sha256: str


@dataclass(frozen=True)
class EvalOptions:
    asr_url: str
    caller_voice: str | None
    caller_rate: int
    response_timeout: float
    silence_timeout: float
    eval_date: str | None
    max_turns: int


@dataclass(frozen=True)
class BatchTask:
    bot: BotSpec
    scenario: dict[str, str]
    batch_id: str
    artifacts_root: Path


class ReservedPort:
    def __init__(self, port: int, sock: socket.socket):
        self.port = port
        self._sock = sock
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._sock.close()
        self._released = True

    def __enter__(self) -> "ReservedPort":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


class PortAllocator:
    def __init__(self, base_port: int = 7860, host: str = "127.0.0.1"):
        self._base_port = base_port
        self._host = host
        self._next_port = base_port
        self._issued: set[int] = set()
        self._lock = threading.Lock()

    def reserve(self) -> ReservedPort:
        with self._lock:
            port = self._next_port
            while True:
                if port in self._issued:
                    port += 1
                    continue
                try:
                    reservation = reserve_localhost_port(port, host=self._host)
                except OSError:
                    port += 1
                    continue
                self._issued.add(port)
                self._next_port = max(self._next_port, port + 1)
                return reservation


def reserve_localhost_port(port: int, host: str = "127.0.0.1") -> ReservedPort:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1)
    except OSError:
        sock.close()
        raise
    return ReservedPort(port, sock)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return slug or "item"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_bot_specs(bot_paths: list[str], *, cwd: Path | None = None) -> list[BotSpec]:
    if cwd is None:
        cwd = Path.cwd()
    cwd = cwd.resolve()

    specs: list[BotSpec] = []
    seen_ids: set[str] = set()
    for raw_path in bot_paths:
        passed_path = str(raw_path)
        candidate = Path(raw_path).expanduser()
        resolved_path = candidate if candidate.is_absolute() else cwd / candidate
        resolved_path = resolved_path.resolve()
        if not resolved_path.exists() or not resolved_path.is_file():
            raise FileNotFoundError(f"Bot file does not exist: {raw_path}")

        name = candidate.stem or resolved_path.stem
        bot_id = slugify(name)
        if bot_id in seen_ids:
            try:
                identity = str(resolved_path.relative_to(cwd).with_suffix(""))
            except ValueError:
                identity = str(resolved_path.with_suffix(""))
            bot_id = slugify(identity)
        if bot_id in seen_ids:
            bot_id = f"{slugify(name)}-{file_sha256(resolved_path)[:8]}"

        seen_ids.add(bot_id)
        specs.append(
            BotSpec(
                id=bot_id,
                name=name,
                passed_path=passed_path,
                resolved_path=resolved_path,
                sha256=file_sha256(resolved_path),
            )
        )

    return specs


def select_scenarios(limit: int | None, ids: list[str] | None) -> list[dict[str, str]]:
    selected = list(SCENARIOS)
    if ids:
        wanted = set(ids)
        selected = [scenario for scenario in selected if scenario["id"] in wanted]
        missing = sorted(wanted - {scenario["id"] for scenario in selected})
        if missing:
            raise ValueError(f"Unknown scenario id(s): {', '.join(missing)}")
    if limit is not None:
        selected = selected[:limit]
    return selected


def default_batch_id(timestamp: float | None = None) -> str:
    if timestamp is None:
        timestamp = time.time()
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(timestamp))}-{uuid.uuid4().hex[:8]}"


def task_artifact_paths(task: BatchTask) -> dict[str, Path]:
    scenario_id = slugify(task.scenario["id"])
    task_dir = task.artifacts_root / task.batch_id / task.bot.id / scenario_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": task_dir,
        "results": task_dir / "results.json",
        "runs": task_dir / "runs.jsonl",
        "latency": task_dir / "latency.jsonl",
        "stage_latency": task_dir / "stage_latency.jsonl",
        "bot_log": task_dir / "bot.log",
        "eval_log": task_dir / "eval.log",
    }


def build_bot_launch_command(bot: BotSpec, port: int) -> list[str]:
    return ["uv", "run", bot.passed_path, "--host", "localhost", "--port", str(port)]


def build_eval_command(
    *,
    scenario_id: str,
    bot_url: str,
    bot: BotSpec,
    batch_id: str,
    paths: dict[str, Path],
    options: EvalOptions,
    bot_command: list[str],
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "eval_runner.py",
        "--mode",
        "voice",
        "--bot-url",
        bot_url,
        "--scenario",
        scenario_id,
        "--results-path",
        str(paths["results"]),
        "--runs-path",
        str(paths["runs"]),
        "--latency-path",
        str(paths["latency"]),
        "--bot-name",
        bot.name,
        "--bot-path",
        bot.passed_path,
        "--batch-id",
        batch_id,
        "--bot-command",
        shlex.join(bot_command),
        "--asr-url",
        options.asr_url,
        "--caller-rate",
        str(options.caller_rate),
        "--response-timeout",
        str(options.response_timeout),
        "--silence-timeout",
        str(options.silence_timeout),
        "--max-turns",
        str(options.max_turns),
    ]
    if options.caller_voice:
        command.extend(["--caller-voice", options.caller_voice])
    if options.eval_date:
        command.extend(["--eval-date", options.eval_date])
    return command


def wait_for_bot_ready(
    bot_url: str,
    *,
    timeout: float,
    process: subprocess.Popen[Any] | None = None,
) -> None:
    status_url = f"{bot_url.rstrip('/')}/status"
    deadline = time.monotonic() + timeout
    last_error = "status endpoint was not polled"

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Bot process exited before ready with code {process.returncode}")
        try:
            request = Request(status_url, method="GET")
            with urlopen(request, timeout=1.5) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except HTTPError as e:
            last_error = f"HTTP {e.code}"
        except (OSError, URLError, TimeoutError) as e:
            last_error = f"{e.__class__.__name__}: {e}"
        time.sleep(0.25)

    raise TimeoutError(
        f"Bot at {bot_url} did not become ready within {timeout:g}s; last error: {last_error}"
    )


def terminate_process(process: subprocess.Popen[Any], *, grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_seconds)


def path_strings(paths: dict[str, Path]) -> dict[str, str]:
    return {key: str(path) for key, path in paths.items()}


def infrastructure_failure_result(
    *,
    task: BatchTask,
    error_type: str,
    reason: str,
    paths: dict[str, Path],
    port: int | None,
    bot_url: str | None,
    wall_time_s: float,
) -> dict[str, Any]:
    scenario_result = {
        "id": task.scenario["id"],
        "passed": False,
        "reason": f"Infrastructure failure ({error_type}): {reason}",
        "turn_count": 0,
        "transcript": [],
        "tool_calls": [],
        "infrastructure_failure": True,
        "error_type": error_type,
        "bot_url": bot_url,
        "port": port,
        "artifacts": path_strings(paths),
    }
    return {
        "bot_id": task.bot.id,
        "scenario_id": task.scenario["id"],
        "scenario": scenario_result,
        "artifacts": path_strings(paths),
        "latency_path": str(paths["latency"]),
        "wall_time_s": round(wall_time_s, 3),
        "port": port,
        "bot_url": bot_url,
        "infrastructure_failure": True,
    }


def load_task_success_result(
    *,
    task: BatchTask,
    paths: dict[str, Path],
    port: int,
    bot_url: str,
    wall_time_s: float,
) -> dict[str, Any]:
    output = json.loads(paths["results"].read_text(encoding="utf-8"))
    scenario_results = output.get("scenarios") or []
    scenario_result = next(
        (result for result in scenario_results if result.get("id") == task.scenario["id"]),
        scenario_results[0] if scenario_results else None,
    )
    if scenario_result is None:
        raise RuntimeError("eval_runner.py produced no scenario results")

    scenario_result = dict(scenario_result)
    scenario_result.update(
        {
            "bot_url": bot_url,
            "port": port,
            "run_id": output.get("run_id"),
            "judge_policy": output.get(
                "judge_policy",
                {"version": JUDGE_POLICY_VERSION, "summary": JUDGE_POLICY_SUMMARY},
            ),
            "voice_agent": output.get("voice_agent"),
            "artifacts": path_strings(paths),
        }
    )
    scenario_result.setdefault("infrastructure_failure", False)
    return {
        "bot_id": task.bot.id,
        "scenario_id": task.scenario["id"],
        "scenario": scenario_result,
        "artifacts": path_strings(paths),
        "latency_path": str(paths["latency"]),
        "wall_time_s": round(wall_time_s, 3),
        "port": port,
        "bot_url": bot_url,
        "infrastructure_failure": False,
    }


def run_batch_task(
    task: BatchTask,
    *,
    options: EvalOptions,
    port_allocator: PortAllocator,
    startup_timeout: float,
    eval_timeout: float,
    cwd: Path = REPO_ROOT,
) -> dict[str, Any]:
    started_at = time.monotonic()
    paths = task_artifact_paths(task)
    port: int | None = None
    bot_url: str | None = None
    process: subprocess.Popen[Any] | None = None

    try:
        reservation = port_allocator.reserve()
        port = reservation.port
        bot_url = f"http://localhost:{port}"
        bot_command = build_bot_launch_command(task.bot, port)
        env = os.environ.copy()
        env["VOICE_LATENCY_LOG_PATH"] = str(paths["latency"])
        env["VOICE_STAGE_LATENCY_LOG_PATH"] = str(paths["stage_latency"])

        with paths["bot_log"].open("w", encoding="utf-8") as bot_log:
            bot_log.write("$ " + shlex.join(bot_command) + "\n")
            bot_log.flush()
            reservation.release()
            process = subprocess.Popen(
                bot_command,
                cwd=cwd,
                stdout=bot_log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            wait_for_bot_ready(bot_url, timeout=startup_timeout, process=process)

            eval_command = build_eval_command(
                scenario_id=task.scenario["id"],
                bot_url=bot_url,
                bot=task.bot,
                batch_id=task.batch_id,
                paths=paths,
                options=options,
                bot_command=bot_command,
            )
            with paths["eval_log"].open("w", encoding="utf-8") as eval_log:
                eval_log.write("$ " + shlex.join(eval_command) + "\n")
                eval_log.flush()
                try:
                    completed = subprocess.run(
                        eval_command,
                        cwd=cwd,
                        stdout=eval_log,
                        stderr=subprocess.STDOUT,
                        timeout=eval_timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired as e:
                    return infrastructure_failure_result(
                        task=task,
                        error_type="eval_timeout",
                        reason=f"eval timed out after {e.timeout:g}s",
                        paths=paths,
                        port=port,
                        bot_url=bot_url,
                        wall_time_s=time.monotonic() - started_at,
                    )

            if completed.returncode != 0:
                return infrastructure_failure_result(
                    task=task,
                    error_type="eval_failed",
                    reason=f"eval_runner.py exited with code {completed.returncode}",
                    paths=paths,
                    port=port,
                    bot_url=bot_url,
                    wall_time_s=time.monotonic() - started_at,
                )
            return load_task_success_result(
                task=task,
                paths=paths,
                port=port,
                bot_url=bot_url,
                wall_time_s=time.monotonic() - started_at,
            )
    except TimeoutError as e:
        return infrastructure_failure_result(
            task=task,
            error_type="startup_timeout",
            reason=str(e),
            paths=paths,
            port=port,
            bot_url=bot_url,
            wall_time_s=time.monotonic() - started_at,
        )
    except Exception as e:
        return infrastructure_failure_result(
            task=task,
            error_type=e.__class__.__name__,
            reason=str(e),
            paths=paths,
            port=port,
            bot_url=bot_url,
            wall_time_s=time.monotonic() - started_at,
        )
    finally:
        if process is not None:
            terminate_process(process)


def metric_value(event: dict[str, Any], key: str) -> int | None:
    value = event.get(key)
    if isinstance(value, int | float):
        return round(value)
    return None


def nearest_rank_percentile(values: list[int], percentile: int) -> int | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = ceil(percentile / 100 * len(sorted_values)) - 1
    return sorted_values[max(index, 0)]


def aggregate_latency(latency_paths: list[str | Path]) -> dict[str, int | None]:
    ttfa: list[int] = []
    ttla: list[int] = []
    completed_turns = 0

    for raw_path in latency_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                status = event.get("status")
                if status is not None and status != "complete":
                    continue

                first_audio = metric_value(event, "ttfa_ms")
                last_audio = metric_value(event, "ttla_ms")
                legacy = metric_value(event, "latency_ms")
                if first_audio is None:
                    first_audio = legacy

                if first_audio is None and last_audio is None:
                    continue

                completed_turns += 1
                if first_audio is not None:
                    ttfa.append(first_audio)
                if last_audio is not None:
                    ttla.append(last_audio)

    return {
        "completed_turns": completed_turns,
        "ttfa_p50_ms": nearest_rank_percentile(ttfa, 50),
        "ttfa_p95_ms": nearest_rank_percentile(ttfa, 95),
        "ttla_p50_ms": nearest_rank_percentile(ttla, 50),
        "ttla_p95_ms": nearest_rank_percentile(ttla, 95),
    }


def aggregate_bot_results(
    *,
    bot: BotSpec,
    task_results: list[dict[str, Any]],
    batch_id: str,
    timestamp: float,
    scenario_order: dict[str, int] | None = None,
) -> dict[str, Any]:
    if scenario_order is None:
        scenario_order = {}

    ordered_results = sorted(
        task_results,
        key=lambda result: scenario_order.get(result["scenario_id"], len(scenario_order)),
    )
    scenarios = [result["scenario"] for result in ordered_results]
    passed_count = sum(1 for scenario in scenarios if bool(scenario.get("passed")))
    scenario_count = len(scenarios)
    judge_error_count = sum(1 for scenario in scenarios if bool(scenario.get("judge_error")))
    infra_failure_count = sum(
        1 for scenario in scenarios if bool(scenario.get("infrastructure_failure"))
    )
    latency = aggregate_latency([result["latency_path"] for result in ordered_results])
    wall_time_s = round(sum(float(result.get("wall_time_s", 0.0)) for result in ordered_results), 3)
    pass_rate = round(passed_count / scenario_count, 3) if scenario_count else 0.0
    artifact_dirs = sorted({result["artifacts"]["dir"] for result in ordered_results})
    artifact_path_keys = ["results", "runs", "latency", "stage_latency", "bot_log", "eval_log"]
    artifact_paths = {
        key: sorted(
            {
                result["artifacts"][key]
                for result in ordered_results
                if key in result["artifacts"]
            }
        )
        for key in artifact_path_keys
    }

    return {
        "batch_id": batch_id,
        "timestamp": timestamp,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        "judge_policy": {
            "version": JUDGE_POLICY_VERSION,
            "summary": JUDGE_POLICY_SUMMARY,
        },
        "bot_id": bot.id,
        "bot_name": bot.name,
        "bot_path": bot.passed_path,
        "bot_resolved_path": str(bot.resolved_path),
        "bot_sha256": bot.sha256,
        "scenario_count": scenario_count,
        "passed_count": passed_count,
        "pass_rate": pass_rate,
        "judge_error_count": judge_error_count,
        "infrastructure_failure_count": infra_failure_count,
        "voice_latency": latency,
        "voice_p95_latency_ms": latency["ttfa_p95_ms"],
        "wall_time_s": wall_time_s,
        "artifact_dirs": artifact_dirs,
        "artifact_paths": artifact_paths,
        "scenarios": scenarios,
    }


def build_batch_output(
    *,
    batch_id: str,
    timestamp: float,
    bots: list[BotSpec],
    selected_scenarios: list[dict[str, str]],
    task_results: list[dict[str, Any]],
    max_workers: int,
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    scenario_order = {scenario["id"]: index for index, scenario in enumerate(selected_scenarios)}
    by_bot: dict[str, list[dict[str, Any]]] = {bot.id: [] for bot in bots}
    for result in task_results:
        by_bot[result["bot_id"]].append(result)

    bot_summaries = {
        bot.id: aggregate_bot_results(
            bot=bot,
            task_results=by_bot[bot.id],
            batch_id=batch_id,
            timestamp=timestamp,
            scenario_order=scenario_order,
        )
        for bot in bots
    }
    total_passed = sum(summary["passed_count"] for summary in bot_summaries.values())
    total_scenarios = sum(summary["scenario_count"] for summary in bot_summaries.values())
    return {
        "batch_id": batch_id,
        "timestamp": timestamp,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        "judge_policy": {
            "version": JUDGE_POLICY_VERSION,
            "summary": JUDGE_POLICY_SUMMARY,
        },
        "bot_count": len(bots),
        "scenario_count": len(selected_scenarios),
        "task_count": len(task_results),
        "max_workers": max_workers,
        "passed_count": total_passed,
        "pass_rate": round(total_passed / total_scenarios, 3) if total_scenarios else 0.0,
        "wall_time_s": round(ended_at - started_at, 3),
        "bot_order": [bot.id for bot in bots],
        "scenario_order": [scenario["id"] for scenario in selected_scenarios],
        "bots": bot_summaries,
    }


def write_batch_results(output: dict[str, Any], path: str | Path = "batch_results.json") -> None:
    results_path = Path(path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def append_batch_runs(output: dict[str, Any], path: str | Path = "batch_runs.jsonl") -> None:
    runs_path = Path(path)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("a", encoding="utf-8") as f:
        for bot_id in output["bot_order"]:
            summary = output["bots"][bot_id]
            row = {
                "batch_id": output["batch_id"],
                "timestamp": output["timestamp"],
                "timestamp_iso": output["timestamp_iso"],
                "judge_policy": output.get("judge_policy"),
                "bot_id": bot_id,
                "bot_name": summary["bot_name"],
                "bot_path": summary["bot_path"],
                "bot_resolved_path": summary["bot_resolved_path"],
                "bot_sha256": summary["bot_sha256"],
                "scenario_count": summary["scenario_count"],
                "passed_count": summary["passed_count"],
                "pass_rate": summary["pass_rate"],
                "judge_error_count": summary["judge_error_count"],
                "infrastructure_failure_count": summary["infrastructure_failure_count"],
                "voice_latency": summary["voice_latency"],
                "voice_p95_latency_ms": summary["voice_p95_latency_ms"],
                "wall_time_s": summary["wall_time_s"],
                "artifact_dirs": summary["artifact_dirs"],
                "artifact_paths": summary["artifact_paths"],
            }
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


BATCH_METRICS_FIELDS = [
    "batch_id",
    "timestamp",
    "timestamp_iso",
    "judge_policy_version",
    "judge_policy_summary",
    "bot_id",
    "bot_name",
    "bot_path",
    "bot_resolved_path",
    "bot_sha256",
    "pass_rate",
    "passed_count",
    "scenario_count",
    "judge_error_count",
    "infrastructure_failure_count",
    "ttfa_p50_ms",
    "ttfa_p95_ms",
    "ttla_p50_ms",
    "ttla_p95_ms",
    "completed_voice_turns",
    "wall_time_s",
    "artifact_dirs",
    "artifact_paths",
]


def append_batch_metrics_csv(
    output: dict[str, Any],
    path: str | Path = "batch_metrics.csv",
) -> None:
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    with metrics_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BATCH_METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        for bot_id in output["bot_order"]:
            summary = output["bots"][bot_id]
            latency = summary["voice_latency"]
            writer.writerow(
                {
                    "batch_id": output["batch_id"],
                    "timestamp": output["timestamp"],
                    "timestamp_iso": output["timestamp_iso"],
                    "judge_policy_version": output.get("judge_policy", {}).get("version"),
                    "judge_policy_summary": output.get("judge_policy", {}).get("summary"),
                    "bot_id": bot_id,
                    "bot_name": summary["bot_name"],
                    "bot_path": summary["bot_path"],
                    "bot_resolved_path": summary["bot_resolved_path"],
                    "bot_sha256": summary["bot_sha256"],
                    "pass_rate": summary["pass_rate"],
                    "passed_count": summary["passed_count"],
                    "scenario_count": summary["scenario_count"],
                    "judge_error_count": summary["judge_error_count"],
                    "infrastructure_failure_count": summary["infrastructure_failure_count"],
                    "ttfa_p50_ms": latency["ttfa_p50_ms"],
                    "ttfa_p95_ms": latency["ttfa_p95_ms"],
                    "ttla_p50_ms": latency["ttla_p50_ms"],
                    "ttla_p95_ms": latency["ttla_p95_ms"],
                    "completed_voice_turns": latency["completed_turns"],
                    "wall_time_s": summary["wall_time_s"],
                    "artifact_dirs": json.dumps(summary["artifact_dirs"], separators=(",", ":")),
                    "artifact_paths": json.dumps(
                        summary["artifact_paths"], separators=(",", ":")
                    ),
                }
            )


def parse_args(argv: list[str] | None = None, *, env_root: Path = REPO_ROOT) -> argparse.Namespace:
    load_batch_environment(env_root)
    parser = argparse.ArgumentParser(description="Run batch voice evals across bot files.")
    parser.add_argument("--bots", nargs="+", required=True, help="Bot files to compare.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum concurrent bot/scenario evals.",
    )
    parser.add_argument("--base-port", type=int, default=7860)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--eval-timeout", type=float, default=600.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenario", action="append", dest="scenario_ids")
    parser.add_argument("--asr-url", default=os.getenv("NVIDIA_ASR_URL", DEFAULT_ASR_URL))
    parser.add_argument("--caller-voice", default=os.getenv("EVAL_CALLER_VOICE"))
    parser.add_argument(
        "--caller-rate",
        type=int,
        default=int(os.getenv("EVAL_CALLER_RATE", "185")),
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=float(os.getenv("EVAL_VOICE_RESPONSE_TIMEOUT", "45")),
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=float(os.getenv("EVAL_VOICE_SILENCE_TIMEOUT", "0.9")),
    )
    parser.add_argument("--eval-date", default=os.getenv("EVAL_TODAY"))
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--artifacts-root", default="batch_artifacts")
    parser.add_argument("--batch-results-path", default="batch_results.json")
    parser.add_argument("--batch-runs-path", default="batch_runs.jsonl")
    parser.add_argument("--batch-metrics-path", default="batch_metrics.csv")
    return parser.parse_args(argv)


def _env_positive_int_or_none(name: str, default: int | None) -> int | None:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return parsed if parsed > 0 else None


def default_worker_count(*, total_tasks: int, bot_count: int) -> int:
    if total_tasks <= 0:
        return 1
    return min(total_tasks, min(6, bot_count * 2))


def resolve_max_workers(
    *,
    requested: int | None,
    total_tasks: int,
    bot_count: int,
) -> tuple[int, int | None, bool]:
    max_workers = (
        requested if requested is not None else default_worker_count(total_tasks=total_tasks, bot_count=bot_count)
    )
    max_workers = max(1, max_workers)

    session_limit = _env_positive_int_or_none(
        "EVAL_TTS_ACTIVE_SESSION_LIMIT",
        DEFAULT_TTS_ACTIVE_SESSION_LIMIT,
    )
    if session_limit is None:
        return max_workers, None, False

    capped_workers = min(max_workers, session_limit)
    return capped_workers, session_limit, capped_workers != max_workers


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_id is None:
        args.batch_id = default_batch_id()
    bots = normalize_bot_specs(args.bots, cwd=REPO_ROOT)
    selected_scenarios = select_scenarios(args.limit, args.scenario_ids)
    tasks = [
        BatchTask(
            bot=bot,
            scenario=scenario,
            batch_id=args.batch_id,
            artifacts_root=Path(args.artifacts_root),
        )
        for bot in bots
        for scenario in selected_scenarios
    ]
    total_tasks = len(tasks)
    max_workers, tts_session_limit, workers_capped = resolve_max_workers(
        requested=args.max_workers,
        total_tasks=total_tasks,
        bot_count=len(bots),
    )

    options = EvalOptions(
        asr_url=args.asr_url,
        caller_voice=args.caller_voice,
        caller_rate=args.caller_rate,
        response_timeout=args.response_timeout,
        silence_timeout=args.silence_timeout,
        eval_date=args.eval_date,
        max_turns=args.max_turns,
    )

    started_at = time.monotonic()
    timestamp = time.time()
    print(
        f"Starting batch {args.batch_id}: {len(bots)} bot(s), "
        f"{len(selected_scenarios)} scenario(s), {max_workers} worker(s).",
        flush=True,
    )
    if workers_capped:
        print(
            "Capped batch workers to "
            f"{max_workers} to stay within EVAL_TTS_ACTIVE_SESSION_LIMIT={tts_session_limit}.",
            flush=True,
        )

    port_allocator = PortAllocator(base_port=args.base_port)
    task_results: list[dict[str, Any]] = []
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    run_batch_task,
                    task,
                    options=options,
                    port_allocator=port_allocator,
                    startup_timeout=args.startup_timeout,
                    eval_timeout=args.eval_timeout,
                )
                for task in tasks
            ]
            for future in as_completed(futures):
                result = future.result()
                task_results.append(result)
                status = "PASS" if result["scenario"].get("passed") else "FAIL"
                print(
                    f"{status} {result['bot_id']} / {result['scenario_id']} "
                    f"({result['wall_time_s']:.1f}s)",
                    flush=True,
                )

    ended_at = time.monotonic()
    output = build_batch_output(
        batch_id=args.batch_id,
        timestamp=timestamp,
        bots=bots,
        selected_scenarios=selected_scenarios,
        task_results=task_results,
        max_workers=max_workers,
        started_at=started_at,
        ended_at=ended_at,
    )
    write_batch_results(output, args.batch_results_path)
    append_batch_runs(output, args.batch_runs_path)
    append_batch_metrics_csv(output, args.batch_metrics_path)
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_id is None:
        args.batch_id = default_batch_id()
    output = run_batch(args)
    print(
        f"{output['passed_count']} task(s) passed across {output['task_count']} task(s) "
        f"({output['pass_rate']:.1%}) - batch {output['batch_id']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
