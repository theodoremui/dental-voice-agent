import json
import os
import subprocess
from pathlib import Path

import batch_eval_runner
from batch_eval_runner import (
    JUDGE_POLICY_VERSION,
    BatchTask,
    BotSpec,
    EvalOptions,
    PortAllocator,
    aggregate_bot_results,
    aggregate_latency,
    build_batch_output,
    build_bot_launch_command,
    build_eval_command,
    file_sha256,
    load_task_success_result,
    normalize_bot_specs,
    resolve_max_workers,
    run_batch_task,
)


def test_parse_args_loads_dotenv_defaults(tmp_path: Path, monkeypatch):
    original_asr_url = os.environ.get("NVIDIA_ASR_URL")
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("NVIDIA_ASR_URL=ws://dotenv-asr.example:8080\n", encoding="utf-8")
    monkeypatch.delenv("NVIDIA_ASR_URL", raising=False)

    try:
        args = batch_eval_runner.parse_args(["--bots", "bot0.py"], env_root=tmp_path)
        assert args.asr_url == "ws://dotenv-asr.example:8080"
    finally:
        if original_asr_url is None:
            os.environ.pop("NVIDIA_ASR_URL", None)
        else:
            os.environ["NVIDIA_ASR_URL"] = original_asr_url


def test_bot_id_path_normalization_and_sha_capture(tmp_path: Path):
    bot_path = tmp_path / "bot0.py"
    bot_path.write_text("print('bot0')\n", encoding="utf-8")

    specs = normalize_bot_specs(["bot0.py"], cwd=tmp_path)

    assert len(specs) == 1
    assert specs[0].id == "bot0"
    assert specs[0].name == "bot0"
    assert specs[0].passed_path == "bot0.py"
    assert specs[0].resolved_path == bot_path.resolve()
    assert specs[0].sha256 == file_sha256(bot_path)


def test_unique_port_reservation_skips_busy_ports(monkeypatch):
    attempts = []
    busy_ports = {7860}

    class FakeSocket:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.port = None

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            port = address[1]
            attempts.append(port)
            if port in busy_ports:
                raise OSError("busy")
            self.port = port

        def listen(self, backlog):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(batch_eval_runner.socket, "socket", FakeSocket)

    allocator = PortAllocator(base_port=7860)
    first = allocator.reserve()
    second = allocator.reserve()

    assert first.port == 7861
    assert second.port == 7862
    assert attempts == [7860, 7861, 7862]
    first.release()
    second.release()


def test_command_construction_for_bot_launch_and_eval_invocation(tmp_path: Path):
    bot_path = tmp_path / "bot1.py"
    bot_path.write_text("print('bot1')\n", encoding="utf-8")
    bot = normalize_bot_specs(["bot1.py"], cwd=tmp_path)[0]
    paths = {
        "results": tmp_path / "results.json",
        "runs": tmp_path / "runs.jsonl",
        "latency": tmp_path / "latency.jsonl",
    }
    options = EvalOptions(
        asr_url="ws://asr.example",
        caller_voice="Samantha",
        caller_rate=170,
        response_timeout=12.5,
        silence_timeout=0.7,
        eval_date="2026-05-30",
        max_turns=4,
    )

    bot_command = build_bot_launch_command(bot, 9001)
    eval_command = build_eval_command(
        scenario_id="scenario_a",
        bot_url="http://localhost:9001",
        bot=bot,
        batch_id="batch-123",
        paths=paths,
        options=options,
        bot_command=bot_command,
    )

    assert bot_command == ["uv", "run", "bot1.py", "--host", "localhost", "--port", "9001"]
    assert eval_command[:9] == [
        "uv",
        "run",
        "python",
        "eval_runner.py",
        "--mode",
        "voice",
        "--bot-url",
        "http://localhost:9001",
        "--scenario",
    ]
    assert "scenario_a" in eval_command
    assert "--bot-name" in eval_command
    assert eval_command[eval_command.index("--bot-path") + 1] == "bot1.py"
    assert eval_command[eval_command.index("--batch-id") + 1] == "batch-123"
    assert eval_command[eval_command.index("--caller-voice") + 1] == "Samantha"


def test_aggregation_builds_bot_level_pass_rates_and_raw_latency(tmp_path: Path):
    bot_path = tmp_path / "bot2.py"
    bot_path.write_text("print('bot2')\n", encoding="utf-8")
    bot = BotSpec(
        id="bot2",
        name="bot2",
        passed_path="bot2.py",
        resolved_path=bot_path,
        sha256=file_sha256(bot_path),
    )
    latency_a = tmp_path / "latency-a.jsonl"
    latency_b = tmp_path / "latency-b.jsonl"
    latency_a.write_text(
        "\n".join(
            [
                json.dumps({"status": "complete", "ttfa_ms": 100, "ttla_ms": 500}),
                json.dumps({"status": "complete", "ttfa_ms": 300, "ttla_ms": 900}),
                json.dumps({"status": "interrupted", "ttfa_ms": 50, "ttla_ms": 50}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    latency_b.write_text(
        json.dumps({"status": "complete", "ttfa_ms": 200, "ttla_ms": 700}) + "\n",
        encoding="utf-8",
    )

    assert aggregate_latency([latency_a, latency_b]) == {
        "completed_turns": 3,
        "ttfa_p50_ms": 200,
        "ttfa_p95_ms": 300,
        "ttla_p50_ms": 700,
        "ttla_p95_ms": 900,
    }

    task_results = [
        {
            "bot_id": "bot2",
            "scenario_id": "scenario_b",
            "scenario": {"id": "scenario_b", "passed": False, "judge_error": True},
            "artifacts": {
                "dir": str(tmp_path / "b"),
                "results": str(tmp_path / "b" / "results.json"),
                "runs": str(tmp_path / "b" / "runs.jsonl"),
                "latency": str(latency_b),
                "bot_log": str(tmp_path / "b" / "bot.log"),
                "eval_log": str(tmp_path / "b" / "eval.log"),
            },
            "latency_path": str(latency_b),
            "wall_time_s": 2.0,
        },
        {
            "bot_id": "bot2",
            "scenario_id": "scenario_a",
            "scenario": {"id": "scenario_a", "passed": True},
            "artifacts": {
                "dir": str(tmp_path / "a"),
                "results": str(tmp_path / "a" / "results.json"),
                "runs": str(tmp_path / "a" / "runs.jsonl"),
                "latency": str(latency_a),
                "bot_log": str(tmp_path / "a" / "bot.log"),
                "eval_log": str(tmp_path / "a" / "eval.log"),
            },
            "latency_path": str(latency_a),
            "wall_time_s": 1.5,
        },
    ]

    summary = aggregate_bot_results(
        bot=bot,
        task_results=task_results,
        batch_id="batch-1",
        timestamp=1_770_000_000,
        scenario_order={"scenario_a": 0, "scenario_b": 1},
    )

    assert summary["passed_count"] == 1
    assert summary["scenario_count"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["judge_error_count"] == 1
    assert summary["judge_policy"]["version"] == JUDGE_POLICY_VERSION
    assert summary["voice_latency"]["ttfa_p95_ms"] == 300
    assert summary["artifact_paths"]["latency"] == [str(latency_a), str(latency_b)]
    assert [scenario["id"] for scenario in summary["scenarios"]] == ["scenario_a", "scenario_b"]


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class FakeReservation:
    port = 9100

    def release(self):
        pass


class FakePortAllocator:
    def reserve(self):
        return FakeReservation()


def make_task(tmp_path: Path) -> BatchTask:
    bot_path = tmp_path / "bot.py"
    bot_path.write_text("print('bot')\n", encoding="utf-8")
    bot = BotSpec(
        id="bot",
        name="bot",
        passed_path=str(bot_path),
        resolved_path=bot_path,
        sha256=file_sha256(bot_path),
    )
    return BatchTask(
        bot=bot,
        scenario={"id": "scenario_a", "persona": "persona", "criteria": "criteria"},
        batch_id="batch-failure",
        artifacts_root=tmp_path / "artifacts",
    )


def make_options() -> EvalOptions:
    return EvalOptions(
        asr_url="ws://asr.example",
        caller_voice=None,
        caller_rate=185,
        response_timeout=45.0,
        silence_timeout=0.9,
        eval_date=None,
        max_turns=8,
    )


def test_failed_startup_records_infrastructure_failure_and_terminates(
    tmp_path: Path, monkeypatch
):
    fake_process = FakeProcess()
    monkeypatch.setattr(batch_eval_runner.subprocess, "Popen", lambda *a, **k: fake_process)
    monkeypatch.setattr(
        batch_eval_runner,
        "wait_for_bot_ready",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("not ready")),
    )

    result = run_batch_task(
        make_task(tmp_path),
        options=make_options(),
        port_allocator=FakePortAllocator(),
        startup_timeout=0.1,
        eval_timeout=1.0,
        cwd=tmp_path,
    )

    assert result["scenario"]["passed"] is False
    assert result["scenario"]["infrastructure_failure"] is True
    assert result["scenario"]["error_type"] == "startup_timeout"
    assert "not ready" in result["scenario"]["reason"]
    assert fake_process.terminated is True


def test_eval_timeout_records_infrastructure_failure_and_continues(tmp_path: Path, monkeypatch):
    fake_process = FakeProcess()
    monkeypatch.setattr(batch_eval_runner.subprocess, "Popen", lambda *a, **k: fake_process)
    monkeypatch.setattr(batch_eval_runner, "wait_for_bot_ready", lambda *a, **k: None)
    monkeypatch.setattr(
        batch_eval_runner.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd="eval", timeout=3)
        ),
    )

    result = run_batch_task(
        make_task(tmp_path),
        options=make_options(),
        port_allocator=FakePortAllocator(),
        startup_timeout=0.1,
        eval_timeout=3.0,
        cwd=tmp_path,
    )

    assert result["scenario"]["passed"] is False
    assert result["scenario"]["infrastructure_failure"] is True
    assert result["scenario"]["error_type"] == "eval_timeout"
    assert fake_process.terminated is True


def test_successful_task_sets_stage_latency_log_path(tmp_path: Path, monkeypatch):
    fake_process = FakeProcess()
    captured_env = {}

    def fake_popen(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return fake_process

    def fake_run(*args, **kwargs):
        results_path = (
            make_task(tmp_path).artifacts_root
            / "batch-failure"
            / "bot"
            / "scenario_a"
            / "results.json"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps(
                {
                    "run_id": "run-success",
                    "scenarios": [{"id": "scenario_a", "passed": True}],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(batch_eval_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(batch_eval_runner, "wait_for_bot_ready", lambda *a, **k: None)
    monkeypatch.setattr(batch_eval_runner.subprocess, "run", fake_run)

    result = run_batch_task(
        make_task(tmp_path),
        options=make_options(),
        port_allocator=FakePortAllocator(),
        startup_timeout=0.1,
        eval_timeout=3.0,
        cwd=tmp_path,
    )

    assert result["scenario"]["passed"] is True
    assert result["artifacts"]["stage_latency"].endswith("/stage_latency.jsonl")
    assert captured_env["VOICE_LATENCY_LOG_PATH"].endswith("/latency.jsonl")
    assert captured_env["VOICE_STAGE_LATENCY_LOG_PATH"].endswith("/stage_latency.jsonl")


def test_load_task_success_result_preserves_eval_infrastructure_failure(tmp_path: Path):
    task = make_task(tmp_path)
    paths = {
        "dir": tmp_path / "task",
        "results": tmp_path / "task" / "results.json",
        "runs": tmp_path / "task" / "runs.jsonl",
        "latency": tmp_path / "task" / "latency.jsonl",
        "bot_log": tmp_path / "task" / "bot.log",
        "eval_log": tmp_path / "task" / "eval.log",
    }
    paths["dir"].mkdir(parents=True)
    paths["results"].write_text(
        json.dumps(
            {
                "run_id": "run-infra",
                "scenarios": [
                    {
                        "id": "scenario_a",
                        "passed": False,
                        "reason": "Voice eval failed before judging.",
                        "infrastructure_failure": True,
                        "error_type": "ConnectionRefusedError",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = load_task_success_result(
        task=task,
        paths=paths,
        port=9100,
        bot_url="http://localhost:9100",
        wall_time_s=1.2,
    )

    assert result["scenario"]["infrastructure_failure"] is True
    assert result["scenario"]["error_type"] == "ConnectionRefusedError"
    assert result["scenario"]["judge_policy"]["version"] == JUDGE_POLICY_VERSION


def test_batch_output_records_lenient_judge_policy(tmp_path: Path):
    bot_path = tmp_path / "bot.py"
    bot_path.write_text("print('bot')\n", encoding="utf-8")
    bot = BotSpec(
        id="bot",
        name="bot",
        passed_path=str(bot_path),
        resolved_path=bot_path,
        sha256=file_sha256(bot_path),
    )
    output = build_batch_output(
        batch_id="batch-policy",
        timestamp=1_770_000_000,
        bots=[bot],
        selected_scenarios=[{"id": "scenario_a", "persona": "persona", "criteria": "criteria"}],
        task_results=[
            {
                "bot_id": "bot",
                "scenario_id": "scenario_a",
                "scenario": {"id": "scenario_a", "passed": True},
                "artifacts": {
                    "dir": str(tmp_path / "task"),
                    "results": str(tmp_path / "task" / "results.json"),
                    "runs": str(tmp_path / "task" / "runs.jsonl"),
                    "latency": str(tmp_path / "task" / "latency.jsonl"),
                    "bot_log": str(tmp_path / "task" / "bot.log"),
                    "eval_log": str(tmp_path / "task" / "eval.log"),
                },
                "latency_path": str(tmp_path / "task" / "latency.jsonl"),
                "wall_time_s": 1.0,
            }
        ],
        max_workers=1,
        started_at=10.0,
        ended_at=11.0,
    )

    assert output["judge_policy"]["version"] == JUDGE_POLICY_VERSION
    assert output["bots"]["bot"]["judge_policy"]["version"] == JUDGE_POLICY_VERSION


def test_resolve_max_workers_caps_gradium_tts_sessions(monkeypatch):
    monkeypatch.delenv("EVAL_TTS_ACTIVE_SESSION_LIMIT", raising=False)

    max_workers, session_limit, capped = resolve_max_workers(
        requested=None,
        total_tasks=10,
        bot_count=2,
    )

    assert max_workers == 3
    assert session_limit == 3
    assert capped is True

    monkeypatch.setenv("EVAL_TTS_ACTIVE_SESSION_LIMIT", "0")
    max_workers, session_limit, capped = resolve_max_workers(
        requested=4,
        total_tasks=10,
        bot_count=2,
    )

    assert max_workers == 4
    assert session_limit is None
    assert capped is False
