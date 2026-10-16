"""GenAI Sustainability Copilot: MiMo (OpenAI-compatible API) with JSON read/write tool calling.

Config via environment:
  MIMO_API_KEY   (required)
  MIMO_BASE_URL  (required) e.g. https://<provider-host>/v1  -- from your MiMo API console
  MIMO_MODEL     model name as shown in your console
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx

from modules.json_tools import TOOL_SCHEMAS, run_tool

MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are the EmissionX Sustainability Copilot for an industrial/vehicular \
emissions platform (Ringelmann opacity, CEMS metrics, CO2 kg/hr, CPCB compliance).
Ground every numeric claim in data you read with the tools; never invent values.
Use list_json_files first if unsure about structure. Prefer query_json for aggregates.
Only use write_json when the user explicitly asks to change data, and say exactly what you changed."""


class ChatUnavailable(Exception):
    pass


def _config() -> Dict[str, str]:
    key = os.getenv("MIMO_API_KEY")
    base = os.getenv("MIMO_BASE_URL")
    model = os.getenv("MIMO_MODEL")
    missing = [n for n, v in (("MIMO_API_KEY", key), ("MIMO_BASE_URL", base), ("MIMO_MODEL", model)) if not v]
    if missing:
        raise ChatUnavailable(f"Set environment variable(s): {', '.join(missing)}")
    return {"key": key, "base": base.rstrip("/"), "model": model}


def _openai_tools(allow_write: bool) -> List[Dict[str, Any]]:
    return [
        {"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in TOOL_SCHEMAS if allow_write or t["name"] != "write_json"
    ]


def chat(message: str, history: Optional[List[Dict[str, Any]]] = None,
         allow_write: bool = True, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run one user turn through the tool loop. history is plain role/content text turns."""
    cfg = _config()
    tools = _openai_tools(allow_write)
    messages: List[Dict[str, Any]] = (
        [{"role": "system", "content": SYSTEM_PROMPT + (
            "\n\nLive dashboard snapshot (JSON):\n" + json.dumps(context) if context else "")}]
        + list(history or []) + [{"role": "user", "content": message}]
    )
    calls: List[Dict[str, Any]] = []

    with httpx.Client(timeout=60.0) as http:
        for _ in range(MAX_TOOL_ROUNDS):
            r = http.post(
                f"{cfg['base']}/chat/completions",
                headers={"Authorization": f"Bearer {cfg['key']}"},
                json={"model": cfg["model"], "messages": messages, "tools": tools,
                      "tool_choice": "auto", "max_tokens": 2048},
            )
            if r.status_code >= 400:
                raise ChatUnavailable(f"MiMo API error {r.status_code}: {r.text[:300]}")
            msg = r.json()["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             **({"tool_calls": tool_calls} if tool_calls else {})})
            if not tool_calls:
                return {"reply": msg.get("content") or "", "tool_calls": calls}

            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "write_json" and not allow_write:
                    out = {"error": "Write access disabled"}
                else:
                    out = run_tool(name, args)
                calls.append({"tool": name, "input": args, "ok": "error" not in out})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(out, ensure_ascii=False)})

    return {"reply": "Stopped: too many tool rounds.", "tool_calls": calls}
