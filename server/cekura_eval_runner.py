import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from scenarios import SCENARIOS

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CEKURA_BASE_URL = "https://api.cekura.ai"
DEFAULT_NAME_PREFIX = "Bright Smile Dental"
SOURCE_TAG = "source:scenarios.py"
SUITE_TAG = "suite:bright-smile-dental"
DEFAULT_METRIC_NAME = "Bright Smile Dental Scenario Outcome"
DEFAULT_METRIC_DESCRIPTION = (
    "Checks whether the Bright Smile Dental main agent satisfied the scenario-specific "
    "success criteria from scenarios.py."
)
DEFAULT_METRIC_PROMPT = """Evaluate whether the Main Agent satisfied the scenario-specific success criteria.

Transcript:
{{transcript}}

Simulation evaluator instructions, when present:
{{evaluator.instructions}}

Observability scenario criteria, when this call was uploaded by cekura_eval_runner.py:
{{dynamic_variables.criteria}}

Observability scenario context, when present:
scenario_id={{dynamic_variables.scenario_id}}
category={{dynamic_variables.category}}
severity={{dynamic_variables.severity}}
persona={{dynamic_variables.persona}}

Pass if the required answer, booking/rescheduling action, confirmation, safe refusal, or
emergency guidance appears anywhere in the Main Agent turns. Fail if the outcome is absent,
materially wrong, unsafe, contradicted, or impossible to infer from the transcript. Return a
binary pass/fail result with a concise explanation."""


class CekuraError(RuntimeError):
    """Raised when Cekura returns an error response or cannot be reached."""


class EvaluatorClient(Protocol):
    def list_evaluators(
        self,
        *,
        agent_id: int | None = None,
        project_id: int | None = None,
        assistant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def create_evaluator(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class ObservabilityClient(Protocol):
    def ingest_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class CekuraClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_CEKURA_BASE_URL,
        timeout: float = 30.0,
        opener: Any = urlopen,
    ):
        if not api_key:
            raise ValueError("CEKURA_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = opener

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> Any:
        url = self._build_url(path, query=query)
        data = None
        headers = {
            "Accept": "application/json",
            "X-CEKURA-API-KEY": self.api_key,
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(url, data=data, headers=headers, method=method.upper())
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise CekuraError(f"Cekura API returned HTTP {e.code}: {body}") from e
        except (OSError, URLError, TimeoutError) as e:
            raise CekuraError(f"Cekura API request failed: {e}") from e

        if not body.strip():
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}

    def _build_url(self, path: str, *, query: dict[str, Any] | None = None) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            url = f"{self.base_url}/{path.lstrip('/')}"
        clean_query = {
            key: value
            for key, value in (query or {}).items()
            if value is not None and value != [] and value != ""
        }
        if clean_query:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{urlencode(clean_query, doseq=True)}"
        return url

    def list_paginated(
        self,
        path: str,
        *,
        query: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        data = self.request("GET", path, query=query)
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []

        rows = list(data.get("results") or [])
        next_url = data.get("next")
        while next_url:
            data = self.request("GET", str(next_url))
            if not isinstance(data, dict):
                break
            rows.extend(data.get("results") or [])
            next_url = data.get("next")
        return rows

    def list_personalities(
        self,
        *,
        project_id: int | None = None,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.list_paginated(
            "/test_framework/v1/personalities/",
            query={"project_id": project_id, "language": language},
        )

    def list_evaluators(
        self,
        *,
        agent_id: int | None = None,
        project_id: int | None = None,
        assistant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.list_paginated(
            "/test_framework/v1/scenarios/",
            query={
                "agent_id": agent_id,
                "project_id": project_id,
                "assistant_id": assistant_id,
                "page_size": 200,
            },
        )

    def create_evaluator(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/test_framework/v1/scenarios/", payload=payload)

    def create_metric(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/test_framework/v1/metrics/", payload=payload)

    def run_text_evaluators(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request(
            "POST",
            "/test_framework/v1/scenarios/run_scenarios_text/",
            payload=payload,
        )

    def get_result(self, result_id: int) -> dict[str, Any]:
        return self.request("GET", f"/test_framework/v1/results/{result_id}/")

    def list_runs(self, run_ids: list[int]) -> list[dict[str, Any]]:
        if not run_ids:
            return []
        data = self.request(
            "GET",
            "/test_framework/v1/runs/bulk/",
            query={"run_ids": ",".join(str(run_id) for run_id in run_ids)},
        )
        return data if isinstance(data, list) else []

    def ingest_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/observability/v1/observe/", payload=payload)


def load_cekura_client(
    *,
    env_root: Path = REPO_ROOT,
    base_url: str = DEFAULT_CEKURA_BASE_URL,
    timeout: float = 30.0,
) -> CekuraClient:
    load_dotenv(env_root / ".env", override=True)
    return CekuraClient(
        os.getenv("CEKURA_API_KEY", ""),
        base_url=base_url,
        timeout=timeout,
    )


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip(".-")
    return slug or "item"


def humanize_scenario_id(scenario_id: str) -> str:
    return scenario_id.replace("_", " ").replace("-", " ").title()


def parse_int_values(values: list[str] | None) -> list[int]:
    parsed: list[int] = []
    for value in values or []:
        for part in value.split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))
    return parsed


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


def scenario_tags(scenario: dict[str, str]) -> list[str]:
    return [
        SUITE_TAG,
        SOURCE_TAG,
        f"scenario:{scenario['id']}",
        f"category:{scenario['category']}",
        f"severity:{scenario['severity']}",
    ]


def scenario_name(scenario: dict[str, str], *, name_prefix: str = DEFAULT_NAME_PREFIX) -> str:
    return f"{name_prefix}: {humanize_scenario_id(scenario['id'])}"


def scenario_instructions(scenario: dict[str, str]) -> str:
    return (
        "You are the Testing Agent, playing the caller in this Bright Smile Dental "
        "evaluation scenario.\n\n"
        f"Scenario id: {scenario['id']}\n"
        f"Category: {scenario['category']}\n"
        f"Severity: {scenario['severity']}\n\n"
        f"Caller persona and goal:\n{scenario['persona']}\n\n"
        f"Main Agent success criteria:\n{scenario['criteria']}\n\n"
        "Stay in character. Provide details only when the scenario says to provide them or "
        "when the Main Agent asks. Do not help the Main Agent satisfy the criteria unless "
        "the caller persona would naturally do so. End the conversation naturally once the "
        "goal is satisfied, refused safely, or clearly impossible."
    )


def scoped_payload(
    *,
    agent_id: int | None = None,
    project_id: int | None = None,
    assistant_id: str | None = None,
    agent_key: str = "agent",
    project_key: str = "project",
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if agent_id is not None:
        payload[agent_key] = agent_id
    if project_id is not None:
        payload[project_key] = project_id
    if assistant_id:
        payload["assistant_id"] = assistant_id
    return payload


def validate_has_scope(
    *,
    agent_id: int | None = None,
    project_id: int | None = None,
    assistant_id: str | None = None,
    allow_assistant: bool = True,
) -> None:
    if agent_id is not None or project_id is not None or (allow_assistant and assistant_id):
        return
    if allow_assistant:
        raise ValueError("Provide at least one of --agent-id, --project-id, or --assistant-id")
    raise ValueError("Provide at least one of --agent-id or --project-id")


def build_evaluator_payload(
    scenario: dict[str, str],
    *,
    personality_id: int,
    agent_id: int | None = None,
    project_id: int | None = None,
    assistant_id: str | None = None,
    metric_ids: list[int] | None = None,
    name_prefix: str = DEFAULT_NAME_PREFIX,
    folder_path: str | None = None,
) -> dict[str, Any]:
    validate_has_scope(agent_id=agent_id, project_id=project_id, assistant_id=assistant_id)
    payload: dict[str, Any] = {
        "name": scenario_name(scenario, name_prefix=name_prefix),
        "personality": personality_id,
        "scenario_type": "instruction",
        "instructions": scenario_instructions(scenario),
        "expected_outcome_prompt": scenario["criteria"],
        "tags": scenario_tags(scenario),
    }
    payload.update(
        scoped_payload(
            agent_id=agent_id,
            project_id=project_id,
            assistant_id=assistant_id,
        )
    )
    if metric_ids:
        payload["metrics"] = metric_ids
    if folder_path:
        payload["folder_path"] = folder_path
    return payload


def build_metric_payload(
    *,
    project_id: int,
    assistant_id: str,
    agent_ids: list[int] | None = None,
    name: str = DEFAULT_METRIC_NAME,
    description: str = DEFAULT_METRIC_DESCRIPTION,
    prompt: str = DEFAULT_METRIC_PROMPT,
    display_order: int = 1,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "description": description,
        "audio_enabled": False,
        "prompt": prompt,
        "project": project_id,
        "assistant_id": assistant_id,
        "type": "llm_judge",
        "eval_type": "binary_workflow_adherence",
        "display_order": display_order,
        "configuration": {},
        "enum_values": [],
        "add_to_new_agents": False,
        "simulation_enabled": True,
        "observability_enabled": True,
        "sampling_enabled": False,
        "evaluation_trigger": "always",
    }
    if agent_ids:
        payload["agents"] = agent_ids
    return payload


def tags_to_set(raw_tags: Any) -> set[str]:
    if isinstance(raw_tags, list):
        return {str(tag) for tag in raw_tags}
    if isinstance(raw_tags, str):
        raw_tags = raw_tags.strip()
        if not raw_tags:
            return set()
        try:
            parsed = json.loads(raw_tags)
        except json.JSONDecodeError:
            return {part.strip() for part in raw_tags.split(",") if part.strip()}
        if isinstance(parsed, list):
            return {str(tag) for tag in parsed}
        return {str(parsed)}
    return set()


def scenario_id_from_evaluator(evaluator: dict[str, Any]) -> str | None:
    for tag in tags_to_set(evaluator.get("tags")):
        if tag.startswith("scenario:"):
            return tag.split(":", 1)[1]
    return None


def index_existing_evaluators(
    evaluators: list[dict[str, Any]],
    *,
    scenarios: list[dict[str, str]],
    name_prefix: str,
) -> dict[str, dict[str, Any]]:
    by_scenario: dict[str, dict[str, Any]] = {}
    scenario_names = {
        scenario_name(scenario, name_prefix=name_prefix): scenario["id"] for scenario in scenarios
    }
    for evaluator in evaluators:
        scenario_id = scenario_id_from_evaluator(evaluator)
        if scenario_id is None:
            scenario_id = scenario_names.get(str(evaluator.get("name", "")))
        if scenario_id and scenario_id not in by_scenario:
            by_scenario[scenario_id] = evaluator
    return by_scenario


def build_sync_plan(
    scenarios: list[dict[str, str]],
    *,
    personality_id: int,
    agent_id: int | None = None,
    project_id: int | None = None,
    assistant_id: str | None = None,
    metric_ids: list[int] | None = None,
    name_prefix: str = DEFAULT_NAME_PREFIX,
    folder_path: str | None = None,
) -> dict[str, Any]:
    return {
        "scenario_count": len(scenarios),
        "payloads": [
            build_evaluator_payload(
                scenario,
                personality_id=personality_id,
                agent_id=agent_id,
                project_id=project_id,
                assistant_id=assistant_id,
                metric_ids=metric_ids,
                name_prefix=name_prefix,
                folder_path=folder_path,
            )
            for scenario in scenarios
        ],
    }


def sync_evaluators(
    client: EvaluatorClient,
    scenarios: list[dict[str, str]],
    *,
    personality_id: int,
    agent_id: int | None = None,
    project_id: int | None = None,
    assistant_id: str | None = None,
    metric_ids: list[int] | None = None,
    name_prefix: str = DEFAULT_NAME_PREFIX,
    folder_path: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    validate_has_scope(agent_id=agent_id, project_id=project_id, assistant_id=assistant_id)
    existing = client.list_evaluators(
        agent_id=agent_id,
        project_id=project_id,
        assistant_id=assistant_id,
    )
    existing_by_scenario = index_existing_evaluators(
        existing,
        scenarios=scenarios,
        name_prefix=name_prefix,
    )
    items: list[dict[str, Any]] = []

    for scenario in scenarios:
        matched = existing_by_scenario.get(scenario["id"])
        payload = build_evaluator_payload(
            scenario,
            personality_id=personality_id,
            agent_id=agent_id,
            project_id=project_id,
            assistant_id=assistant_id,
            metric_ids=metric_ids,
            name_prefix=name_prefix,
            folder_path=folder_path,
        )
        if matched:
            items.append(
                {
                    "scenario_id": scenario["id"],
                    "status": "exists",
                    "evaluator_id": matched.get("id"),
                    "name": matched.get("name"),
                }
            )
            continue

        if dry_run:
            items.append(
                {
                    "scenario_id": scenario["id"],
                    "status": "would_create",
                    "payload": payload,
                }
            )
            continue

        created = client.create_evaluator(payload)
        items.append(
            {
                "scenario_id": scenario["id"],
                "status": "created",
                "evaluator_id": created.get("id"),
                "name": created.get("name"),
            }
        )

    return {
        "scenario_count": len(scenarios),
        "created_count": sum(1 for item in items if item["status"] == "created"),
        "existing_count": sum(1 for item in items if item["status"] == "exists"),
        "would_create_count": sum(1 for item in items if item["status"] == "would_create"),
        "items": items,
    }


def resolve_synced_evaluator_ids(
    client: EvaluatorClient,
    scenarios: list[dict[str, str]],
    *,
    agent_id: int | None = None,
    project_id: int | None = None,
    assistant_id: str | None = None,
    name_prefix: str = DEFAULT_NAME_PREFIX,
) -> list[int]:
    existing = client.list_evaluators(
        agent_id=agent_id,
        project_id=project_id,
        assistant_id=assistant_id,
    )
    existing_by_scenario = index_existing_evaluators(
        existing,
        scenarios=scenarios,
        name_prefix=name_prefix,
    )
    missing = [scenario["id"] for scenario in scenarios if scenario["id"] not in existing_by_scenario]
    if missing:
        raise ValueError(
            "Missing Cekura evaluator(s) for scenario id(s): "
            f"{', '.join(missing)}. Run the sync command first."
        )

    evaluator_ids: list[int] = []
    for scenario in scenarios:
        evaluator_id = existing_by_scenario[scenario["id"]].get("id")
        if evaluator_id is None:
            raise ValueError(f"Cekura evaluator for {scenario['id']} has no id")
        evaluator_ids.append(int(evaluator_id))
    return evaluator_ids


def run_text_evaluators(
    client: CekuraClient,
    *,
    evaluator_ids: list[int],
    agent_id: int | None = None,
    project_id: int | None = None,
    assistant_id: str | None = None,
    name: str | None = None,
    frequency: int = 1,
    websocket_url: str | None = None,
    concurrency_limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not evaluator_ids:
        raise ValueError("At least one evaluator id is required")
    validate_has_scope(agent_id=agent_id, project_id=project_id, assistant_id=assistant_id)
    payload: dict[str, Any] = {
        "name": name or f"Bright Smile Dental text run {utc_now_iso()}",
        "scenarios": evaluator_ids,
        "frequency": frequency,
    }
    payload.update(
        scoped_payload(
            agent_id=agent_id,
            project_id=project_id,
            assistant_id=assistant_id,
            agent_key="agent_id",
            project_key="project_id",
        )
    )
    if websocket_url:
        payload["websocket_url"] = websocket_url
    if concurrency_limit is not None:
        payload["concurrency_limit"] = concurrency_limit
    if dry_run:
        return {"status": "dry_run", "payload": payload}
    return client.run_text_evaluators(payload)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def transcript_to_cekura(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = 0.0
    for turn in transcript:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        speaker = str(turn.get("speaker") or "").lower()
        role = "Main Agent" if speaker == "agent" else "Testing Agent"
        duration = max(0.8, min(12.0, len(text.split()) * 0.35))
        start_time = round(cursor, 3)
        end_time = round(cursor + duration, 3)
        rows.append(
            {
                "role": role,
                "content": text,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        cursor = end_time + 0.2
    return rows


def infer_call_end_reason(transcript: list[dict[str, Any]]) -> str:
    for turn in reversed(transcript):
        speaker = str(turn.get("speaker") or "").lower()
        if speaker == "agent":
            return "agent-ended-call"
        if speaker == "caller":
            return "customer-ended-call"
    return "customer-ended-call"


def stable_call_id(record: dict[str, Any]) -> str:
    scenario = record["scenario"]
    scenario_id = str(scenario.get("id") or "scenario")
    bot_id = str(record.get("bot_id") or "bot")
    run_id = str(scenario.get("run_id") or record.get("run_id") or record.get("batch_id") or "run")
    raw = f"bsd-{bot_id}-{scenario_id}-{run_id}"
    slug = slugify(raw)
    if len(slug) <= 100:
        return slug
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return slugify(f"bsd-{scenario_id[:60]}-{digest}")[:100]


def scenario_lookup() -> dict[str, dict[str, str]]:
    return {scenario["id"]: scenario for scenario in SCENARIOS}


def iter_local_result_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(data.get("bots"), dict):
        for bot_id, summary in data["bots"].items():
            for scenario in summary.get("scenarios") or []:
                records.append(
                    {
                        "scenario": scenario,
                        "bot_id": bot_id,
                        "bot_name": summary.get("bot_name"),
                        "run_id": scenario.get("run_id") or data.get("batch_id"),
                        "batch_id": data.get("batch_id"),
                        "timestamp_iso": data.get("timestamp_iso"),
                        "source": "batch_results",
                    }
                )
        return records

    for scenario in data.get("scenarios") or []:
        voice_agent = data.get("voice_agent") or {}
        records.append(
            {
                "scenario": scenario,
                "bot_id": voice_agent.get("name") or "local",
                "bot_name": voice_agent.get("name"),
                "run_id": data.get("run_id"),
                "batch_id": voice_agent.get("batch_id") or data.get("batch_id"),
                "timestamp_iso": data.get("timestamp_iso"),
                "source": "results",
            }
        )
    return records


def build_observability_payload(
    record: dict[str, Any],
    *,
    agent_id: int | None = None,
    assistant_id: str | None = None,
    metric_ids: list[int] | None = None,
) -> dict[str, Any] | None:
    if agent_id is None and not assistant_id:
        raise ValueError("Provide --agent-id or --assistant-id for observability ingestion")

    scenario_result = record["scenario"]
    transcript = scenario_result.get("transcript") or []
    if not isinstance(transcript, list):
        return None

    transcript_json = transcript_to_cekura(transcript)
    if not transcript_json:
        return None

    scenario_def = scenario_lookup().get(str(scenario_result.get("id")), {})
    dynamic_variables = {
        "scenario_id": scenario_result.get("id"),
        "category": scenario_result.get("category") or scenario_def.get("category"),
        "severity": scenario_def.get("severity"),
        "persona": scenario_def.get("persona"),
        "criteria": scenario_def.get("criteria"),
        "local_passed": bool(scenario_result.get("passed")),
        "local_reason": scenario_result.get("reason"),
        "bot_id": record.get("bot_id"),
        "bot_name": record.get("bot_name"),
        "batch_id": record.get("batch_id"),
    }
    metadata = {
        "source": "cekura_eval_runner.py",
        "source_result_type": record.get("source"),
        "scenario_id": scenario_result.get("id"),
        "category": dynamic_variables["category"],
        "severity": dynamic_variables["severity"],
        "local_passed": dynamic_variables["local_passed"],
        "local_reason": dynamic_variables["local_reason"],
        "bot_id": record.get("bot_id"),
        "batch_id": record.get("batch_id"),
    }
    payload: dict[str, Any] = {
        "call_id": stable_call_id(record),
        "transcript_type": "cekura",
        "transcript_json": transcript_json,
        "call_ended_reason": infer_call_end_reason(transcript),
        "customer_number": f"scenario:{scenario_result.get('id')}",
        "metadata": metadata,
        "dynamic_variables": dynamic_variables,
        "timestamp": record.get("timestamp_iso") or utc_now_iso(),
        "feedback": (
            f"Local harness verdict: {'PASS' if scenario_result.get('passed') else 'FAIL'} - "
            f"{scenario_result.get('reason', '')}"
        )[:1000],
    }
    if agent_id is not None:
        payload["agent"] = agent_id
    if assistant_id:
        payload["assistant_id"] = assistant_id
    if metric_ids:
        payload["metric_ids"] = ",".join(str(metric_id) for metric_id in metric_ids)
    return payload


def ingest_local_results(
    client: ObservabilityClient,
    *,
    results_path: Path,
    agent_id: int | None = None,
    assistant_id: str | None = None,
    metric_ids: list[int] | None = None,
    scenario_ids: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    records = iter_local_result_records(data)
    if scenario_ids:
        wanted = set(scenario_ids)
        records = [record for record in records if record["scenario"].get("id") in wanted]
    if limit is not None:
        records = records[:limit]

    items: list[dict[str, Any]] = []
    for record in records:
        payload = build_observability_payload(
            record,
            agent_id=agent_id,
            assistant_id=assistant_id,
            metric_ids=metric_ids,
        )
        if payload is None:
            items.append(
                {
                    "scenario_id": record["scenario"].get("id"),
                    "status": "skipped",
                    "reason": "scenario had no transcript",
                }
            )
            continue
        if dry_run:
            items.append(
                {
                    "scenario_id": record["scenario"].get("id"),
                    "status": "would_ingest",
                    "payload": payload,
                }
            )
            continue
        response = client.ingest_call(payload)
        items.append(
            {
                "scenario_id": record["scenario"].get("id"),
                "status": "ingested",
                "call_id": response.get("call_id") or payload["call_id"],
                "call_log_id": response.get("id"),
            }
        )

    return {
        "results_path": str(results_path),
        "record_count": len(records),
        "ingested_count": sum(1 for item in items if item["status"] == "ingested"),
        "would_ingest_count": sum(1 for item in items if item["status"] == "would_ingest"),
        "skipped_count": sum(1 for item in items if item["status"] == "skipped"),
        "items": items,
    }


def write_or_print_json(output: dict[str, Any] | list[dict[str, Any]], path: str | None) -> None:
    text = json.dumps(output, indent=2) + "\n"
    if path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def add_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-id", type=int)
    parser.add_argument("--project-id", type=int)
    parser.add_argument("--assistant-id")


def add_scenario_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", action="append", dest="scenario_ids")
    parser.add_argument("--limit", type=int)


def add_metric_ids_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metric-id",
        action="append",
        dest="metric_id_values",
        help="Metric ID to attach/use. Repeat or pass comma-separated IDs.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync and run Bright Smile Dental scenarios through Cekura."
    )
    parser.add_argument("--base-url", default=DEFAULT_CEKURA_BASE_URL)
    parser.add_argument("--timeout", type=float, default=30.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_personalities = subparsers.add_parser("list-personalities")
    list_personalities.add_argument("--project-id", type=int)
    list_personalities.add_argument("--language")
    list_personalities.add_argument("--output")

    create_metric = subparsers.add_parser("create-metric")
    create_metric.add_argument("--project-id", type=int, required=True)
    create_metric.add_argument("--assistant-id", required=True)
    create_metric.add_argument("--agent-id", action="append", dest="agent_id_values")
    create_metric.add_argument("--name", default=DEFAULT_METRIC_NAME)
    create_metric.add_argument("--description", default=DEFAULT_METRIC_DESCRIPTION)
    create_metric.add_argument("--prompt-path")
    create_metric.add_argument("--display-order", type=int, default=1)
    create_metric.add_argument("--dry-run", action="store_true")
    create_metric.add_argument("--output")

    plan = subparsers.add_parser("plan")
    add_scope_args(plan)
    add_scenario_filter_args(plan)
    add_metric_ids_arg(plan)
    plan.add_argument("--personality-id", type=int, required=True)
    plan.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    plan.add_argument("--folder-path")
    plan.add_argument("--output")

    sync = subparsers.add_parser("sync")
    add_scope_args(sync)
    add_scenario_filter_args(sync)
    add_metric_ids_arg(sync)
    sync.add_argument("--personality-id", type=int, required=True)
    sync.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    sync.add_argument("--folder-path")
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("--output")

    run_text = subparsers.add_parser("run-text")
    add_scope_args(run_text)
    add_scenario_filter_args(run_text)
    run_text.add_argument(
        "--evaluator-id",
        action="append",
        dest="evaluator_id_values",
        help="Cekura evaluator ID. Repeat or pass comma-separated IDs.",
    )
    run_text.add_argument("--name")
    run_text.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    run_text.add_argument("--frequency", type=int, default=1)
    run_text.add_argument("--websocket-url")
    run_text.add_argument("--concurrency-limit", type=int)
    run_text.add_argument("--dry-run", action="store_true")
    run_text.add_argument("--output")

    get_result = subparsers.add_parser("get-result")
    get_result.add_argument("result_id", type=int)
    get_result.add_argument("--output")

    list_runs_parser = subparsers.add_parser("list-runs")
    list_runs_parser.add_argument("run_ids", nargs="+")
    list_runs_parser.add_argument("--output")

    ingest = subparsers.add_parser("ingest-results")
    ingest.add_argument("--results-path", type=Path, default=Path("results.json"))
    ingest.add_argument("--agent-id", type=int)
    ingest.add_argument("--assistant-id")
    add_scenario_filter_args(ingest)
    add_metric_ids_arg(ingest)
    ingest.add_argument("--dry-run", action="store_true")
    ingest.add_argument("--output")

    return parser


def run_command(args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    if args.command == "create-metric":
        agent_ids = parse_int_values(args.agent_id_values)
        prompt = DEFAULT_METRIC_PROMPT
        if args.prompt_path:
            prompt = Path(args.prompt_path).read_text(encoding="utf-8")
        payload = build_metric_payload(
            project_id=args.project_id,
            assistant_id=args.assistant_id,
            agent_ids=agent_ids,
            name=args.name,
            description=args.description,
            prompt=prompt,
            display_order=args.display_order,
        )
        if args.dry_run:
            return {"status": "dry_run", "payload": payload}
        client = load_cekura_client(
            base_url=args.base_url,
            timeout=args.timeout,
        )
        return client.create_metric(payload)

    if args.command == "plan":
        scenarios = select_scenarios(args.limit, args.scenario_ids)
        return build_sync_plan(
            scenarios,
            personality_id=args.personality_id,
            agent_id=args.agent_id,
            project_id=args.project_id,
            assistant_id=args.assistant_id,
            metric_ids=parse_int_values(args.metric_id_values),
            name_prefix=args.name_prefix,
            folder_path=args.folder_path,
        )

    client = load_cekura_client(
        base_url=args.base_url,
        timeout=args.timeout,
    )

    if args.command == "list-personalities":
        return client.list_personalities(project_id=args.project_id, language=args.language)

    if args.command == "sync":
        scenarios = select_scenarios(args.limit, args.scenario_ids)
        return sync_evaluators(
            client,
            scenarios,
            personality_id=args.personality_id,
            agent_id=args.agent_id,
            project_id=args.project_id,
            assistant_id=args.assistant_id,
            metric_ids=parse_int_values(args.metric_id_values),
            name_prefix=args.name_prefix,
            folder_path=args.folder_path,
            dry_run=args.dry_run,
        )

    if args.command == "run-text":
        evaluator_ids = parse_int_values(args.evaluator_id_values)
        if not evaluator_ids:
            scenarios = select_scenarios(args.limit, args.scenario_ids)
            evaluator_ids = resolve_synced_evaluator_ids(
                client,
                scenarios,
                agent_id=args.agent_id,
                project_id=args.project_id,
                assistant_id=args.assistant_id,
                name_prefix=args.name_prefix,
            )
        return run_text_evaluators(
            client,
            evaluator_ids=evaluator_ids,
            agent_id=args.agent_id,
            project_id=args.project_id,
            assistant_id=args.assistant_id,
            name=args.name,
            frequency=args.frequency,
            websocket_url=args.websocket_url,
            concurrency_limit=args.concurrency_limit,
            dry_run=args.dry_run,
        )

    if args.command == "get-result":
        return client.get_result(args.result_id)

    if args.command == "list-runs":
        return client.list_runs(parse_int_values(args.run_ids))

    if args.command == "ingest-results":
        return ingest_local_results(
            client,
            results_path=args.results_path,
            agent_id=args.agent_id,
            assistant_id=args.assistant_id,
            metric_ids=parse_int_values(args.metric_id_values),
            scenario_ids=args.scenario_ids,
            limit=args.limit,
            dry_run=args.dry_run,
        )

    raise ValueError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = run_command(args)
    except (CekuraError, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    write_or_print_json(output, getattr(args, "output", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
