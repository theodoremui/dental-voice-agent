from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

import tools as tool_module
from tools import TOOL_IMPLS, TOOLS, PipecatToolResult, build_system_instruction

load_dotenv(override=True)

SESSIONS: dict[str, TextBotSession] = {}
ACTIVE_SESSION_ID: str | None = None
SESSION_LOCK = threading.Lock()


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _parse_today(value: str | None) -> date:
    if not value:
        return date.today()
    return date.fromisoformat(value)


def _safe_json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {"_raw": value}
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def _content_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        value = part.get("text") or part.get("content")
        return value if isinstance(value, str) else ""
    value = getattr(part, "text", None) or getattr(part, "content", None)
    return value if isinstance(value, str) else ""


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(_content_part_text(part) for part in content).strip()

    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = model_extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _chat_completion(client: OpenAI, *, disable_thinking: bool, **kwargs: Any):
    if not disable_thinking:
        return client.chat.completions.create(**kwargs)

    try:
        return client.chat.completions.create(
            **kwargs,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except BadRequestError as exc:
        message = str(exc).lower()
        if "chat_template_kwargs" in message or "extra" in message or "unknown" in message:
            return client.chat.completions.create(**kwargs)
        raise


def _reset_tool_state(initial_bookings: list[Mapping[str, Any]]) -> None:
    tool_module._BOOKINGS.clear()
    tool_module._next_id[0] = 1000

    for booking in initial_bookings:
        confirmation_id = str(booking["confirmation_id"])
        tool_module._BOOKINGS[confirmation_id] = {
            "name": booking.get("name"),
            "date": booking.get("date"),
            "time": booking.get("time"),
            "reason": booking.get("reason"),
        }
        match = re.fullmatch(r"BSD(\d+)", confirmation_id)
        if match:
            tool_module._next_id[0] = max(tool_module._next_id[0], int(match.group(1)))


def _run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOL_IMPLS:
        return {"error": f"unknown_tool:{name}", "args": args}

    result = TOOL_IMPLS[name](args)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    if isinstance(result, PipecatToolResult):
        return result.value
    if isinstance(result, dict):
        return result
    return {"value": result}


class TextBotSession:
    def __init__(self, *, from_number: str | None, today: date):
        self.model = _first_env(
            "NEMOTRON_LLM_MODEL",
            "EVAL_AGENT_MODEL",
            "EVAL_MODEL",
            default="nvidia/nemotron-3-super",
        ) or "nvidia/nemotron-3-super"
        self.client = OpenAI(
            api_key=_first_env(
                "NEMOTRON_LLM_API_KEY",
                "EVAL_AGENT_API_KEY",
                "EVAL_API_KEY",
                "NIM_API_KEY",
                "NVIDIA_API_KEY",
                default="EMPTY",
            ),
            base_url=_first_env(
                "NEMOTRON_LLM_URL",
                "EVAL_AGENT_BASE_URL",
                "EVAL_BASE_URL",
                "NIM_BASE_URL",
                "NVIDIA_BASE_URL",
                default="http://192.168.7.228:8000/v1",
            )
            or "",
        )
        self.temperature = _env_float("EVAL_AGENT_TEMPERATURE", 0.3)
        self.disable_thinking = _env_bool("EVAL_DISABLE_THINKING", True)
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_instruction(from_number=from_number, today=today),
            }
        ]

    def reply(self, caller_text: str, *, max_tool_rounds: int = 6) -> dict[str, Any]:
        started_at = time.perf_counter()
        self.messages.append({"role": "user", "content": caller_text})
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        ended = False

        for _ in range(max_tool_rounds):
            response = _chat_completion(
                self.client,
                disable_thinking=self.disable_thinking,
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                temperature=self.temperature,
                max_tokens=360,
            )
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))

            content = _message_text(message)
            if content:
                text_parts.append(content)

            if not message.tool_calls:
                break

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = _safe_json_loads(tool_call.function.arguments)
                result = _run_tool(name, args)
                ended = ended or name == "end_call"
                tool_calls.append({"name": name, "args": args, "result": result})
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

            if ended and text_parts:
                break

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return {
            "text": " ".join(part for part in text_parts if part).strip(),
            "tool_calls": tool_calls,
            "ended": ended,
            "elapsed_ms": elapsed_ms,
        }


class EvalHandler(BaseHTTPRequestHandler):
    server_version = "DentalBotEval/1.0"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json({"ok": True})
            return
        self._write_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/eval/reset":
                self._handle_reset(payload)
                return
            if self.path == "/eval/reply":
                self._handle_reply(payload)
                return
            self._write_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_reset(self, payload: Mapping[str, Any]) -> None:
        global ACTIVE_SESSION_ID

        initial_bookings = payload.get("initial_bookings") or []
        if not isinstance(initial_bookings, list):
            raise ValueError("initial_bookings must be a list")

        with SESSION_LOCK:
            _reset_tool_state(initial_bookings)
            session = TextBotSession(
                from_number=payload.get("from_number"),
                today=_parse_today(payload.get("today")),
            )
            session_id = uuid.uuid4().hex
            ACTIVE_SESSION_ID = session_id
            SESSIONS[session_id] = session
        self._write_json({"session_id": session_id, "model": session.model})

    def _handle_reply(self, payload: Mapping[str, Any]) -> None:
        session_id = payload.get("session_id")
        caller_text = payload.get("caller_text")
        if not isinstance(caller_text, str):
            raise ValueError("caller_text must be a string")
        with SESSION_LOCK:
            if not isinstance(session_id, str) or session_id not in SESSIONS:
                self._write_json({"error": "unknown_session"}, status=HTTPStatus.NOT_FOUND)
                return
            if session_id != ACTIVE_SESSION_ID:
                self._write_json(
                    {"error": "session_superseded"},
                    status=HTTPStatus.CONFLICT,
                )
                return
            self._write_json(SESSIONS[session_id].reply(caller_text))

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        loaded = json.loads(raw.decode("utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("request body must be a JSON object")
        return loaded

    def _write_json(self, payload: Mapping[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    host = os.getenv("BOT_EVAL_HOST", "127.0.0.1")
    port = int(os.getenv("BOT_EVAL_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), EvalHandler)
    print(f"bot.py eval server listening on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
