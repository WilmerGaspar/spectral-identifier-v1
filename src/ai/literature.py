"""
Literature grounding with Valency MCP Bond
==========================================
After the spectral analysis identifies phases (and possibly unexplained,
unknown peaks), this module asks Valency's MCP Bond server for research
papers about them, so every result in the report comes with citations.

Connection (from https://docs.valency.io/guides/api/):
  * endpoint  https://mcp.valency.io/   (MCP, Streamable HTTP, JSON-RPC)
  * header    Authorization: Bearer <token>   (token from https://app.valency.io/settings)

The token is read from the VALENCY_TOKEN environment variable. NEVER commit
it to the repository.

The search tool's argument names are not documented on the public page, so
they are read at run time from the server's own tool schema (tools/list).

Usage:
    set VALENCY_TOKEN=...            (Windows)   /  export VALENCY_TOKEN=...  (Linux/Mac)
    python -m src.ai.literature Grossular Quartz --unknown 1008
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import requests

ENDPOINT = "https://mcp.valency.io/"
PROTOCOL_VERSION = "2025-11-25"
SEARCH_TOOL = "semantic_search_papers"


class MCPError(RuntimeError):
    pass


class ValencyClient:
    """Minimal MCP Streamable-HTTP client (initialize, tools/list, tools/call)."""

    def __init__(self, token: str, endpoint: str = ENDPOINT, timeout: int = 60):
        if not token:
            raise MCPError("Missing token: set the VALENCY_TOKEN environment variable.")
        self.endpoint = endpoint
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self.protocol = PROTOCOL_VERSION
        self._id = 0
        self.headers = {"Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream"}

    # transport ------------------------------------------------
    def _post(self, payload: dict, expect_reply: bool = True) -> Optional[dict]:
        headers = dict(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if payload.get("method") != "initialize":
            headers["MCP-Protocol-Version"] = self.protocol
        r = requests.post(self.endpoint, headers=headers, data=json.dumps(payload),
                          timeout=self.timeout, stream=True)
        if r.status_code >= 400:
            raise MCPError(f"HTTP {r.status_code}: {r.text[:300]}")
        sid = r.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid
        if not expect_reply:
            return None
        ctype = r.headers.get("Content-Type", "")
        if "text/event-stream" in ctype:
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    msg = json.loads(line[5:].strip())
                    if msg.get("id") == payload.get("id"):
                        return self._check(msg)
            raise MCPError("No reply in event stream.")
        return self._check(r.json())

    @staticmethod
    def _check(msg: dict) -> dict:
        if "error" in msg:
            raise MCPError(f"MCP error: {msg['error']}")
        return msg.get("result", {})

    def _request(self, method: str, params: Optional[dict] = None) -> dict:
        self._id += 1
        return self._post({"jsonrpc": "2.0", "id": self._id, "method": method,
                           "params": params or {}})

    # MCP ------------------------------------------------------
    def connect(self) -> dict:
        res = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "spectral-identifier", "version": "1.0"}})
        self.protocol = res.get("protocolVersion", PROTOCOL_VERSION)
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, expect_reply=False)
        return res

    def list_tools(self) -> List[dict]:
        return self._request("tools/list").get("tools", [])

    def call(self, name: str, arguments: dict) -> dict:
        return self._request("tools/call", {"name": name, "arguments": arguments})


def _query_args(tool: dict, query: str, limit: int) -> dict:
    """Map our query/limit onto the tool's own input schema."""
    props = (tool.get("inputSchema") or {}).get("properties", {})
    required = (tool.get("inputSchema") or {}).get("required", [])
    str_keys = [k for k, v in props.items() if v.get("type") == "string"]
    preferred = [k for k in ("query", "q", "text", "search_query") if k in props]
    qkey = (preferred or [k for k in required if k in str_keys] or str_keys or ["query"])[0]
    args = {qkey: query}
    for k, v in props.items():
        if v.get("type") == "integer" and any(w in k.lower() for w in ("limit", "top", "max", "num", "k")):
            args[k] = limit
            break
    return args


def _content_text(result: dict) -> str:
    parts = []
    for c in result.get("content", []):
        if c.get("type") == "text":
            parts.append(c.get("text", ""))
    return "\n".join(parts) if parts else json.dumps(result.get("structuredContent", result))[:4000]


def build_queries(phases: List[str], unknown_peaks_cm: List[float],
                  technique: str = "Raman") -> Dict[str, str]:
    q = {}
    for p in phases:
        q[f"phase: {p}"] = f"{technique} spectrum of {p} band assignment"
    for x in unknown_peaks_cm:
        q[f"unknown peak: {x:.0f} cm-1"] = f"mineral {technique} band near {x:.0f} cm-1 assignment"
    return q


def literature_report(phases: List[str], unknown_peaks_cm: List[float], token: str,
                      limit: int = 3, endpoint: str = ENDPOINT) -> str:
    client = ValencyClient(token, endpoint)
    client.connect()
    tools = {t["name"]: t for t in client.list_tools()}
    if SEARCH_TOOL not in tools:
        raise MCPError(f"'{SEARCH_TOOL}' not offered by the server; tools: {sorted(tools)[:10]}")
    out = ["# Literature for this analysis (Valency MCP Bond)\n"]
    for label, query in build_queries(phases, unknown_peaks_cm).items():
        res = client.call(SEARCH_TOOL, _query_args(tools[SEARCH_TOOL], query, limit))
        out.append(f"## {label}\n_query: {query}_\n\n{_content_text(res)}\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Find papers for identified phases with Valency MCP Bond")
    ap.add_argument("phases", nargs="*", help="identified minerals, e.g. Grossular Quartz")
    ap.add_argument("--unknown", nargs="*", type=float, default=[], help="unexplained peaks (cm-1)")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--endpoint", default=ENDPOINT)
    a = ap.parse_args()
    try:
        print(literature_report(a.phases, a.unknown, os.environ.get("VALENCY_TOKEN", ""),
                                a.limit, a.endpoint))
    except MCPError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
