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
    python -m src.ai.literature Zircon Calcite --topic xrd      # structure / unit-cell papers
    python -m src.ai.literature Zircon --json                   # structured output (for an LLM)
    python -m src.ai.literature --tools                         # what the server offers

A paper is kept as evidence only if the mineral's name appears in its title or
abstract; semantic search alone can return look-alike minerals.
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
                  topic: str = "raman") -> Dict[str, str]:
    """One search query per identified phase (and per unexplained Raman peak)."""
    q = {}
    for p in phases:
        if topic == "xrd":
            q[p] = f"{p} crystal structure refinement unit cell X-ray diffraction"
        else:
            q[p] = f"Raman spectrum of {p} band assignment"
    for x in unknown_peaks_cm:
        q[f"unknown peak {x:.0f} cm-1"] = f"mineral Raman band near {x:.0f} cm-1 assignment"
    return q


def _mineral_stem(name: str) -> str:
    """'Xenotime-(Y)' -> 'xenotime'; used to check that a paper is about the mineral."""
    return name.split("-(")[0].split("(")[0].strip().lower()


def _parse_search(text: str) -> dict:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"papers": []}
    except (json.JSONDecodeError, TypeError):
        return {"papers": [], "raw": text}


def check_papers(label: str, data: dict) -> dict:
    """
    Valency's search is semantic, so for less common minerals it can return
    papers about a DIFFERENT mineral with a similar-sounding description
    (e.g. 'sidorenkite' for 'siderite'). We keep a paper as evidence only if
    the mineral's name appears in its title or abstract.
    """
    stem = None if label.startswith("unknown peak") else _mineral_stem(label)
    papers = []
    for pp in data.get("papers", []):
        text = f"{pp.get('title', '')} {pp.get('abstract', '')}".lower()
        first = ((pp.get("dates") or {}).get("first_available") or {}).get("value", "")
        year = (first or pp.get("first_submitted") or pp.get("datestamp") or "")[:4]
        papers.append({
            "title": (pp.get("title") or "").strip().rstrip("."),
            "year": year,
            "authors": pp.get("authors") or [],
            "doi": pp.get("doi") or "",
            "url": pp.get("url") or "",
            "score": pp.get("relevance_score", pp.get("final_score")),
            "mentions_phase": True if stem is None else stem in text,
        })
    papers.sort(key=lambda x: (not x["mentions_phase"], -(x["score"] or 0)))
    return {"label": label, "confidence": data.get("confidence", ""),
            "warnings": data.get("warnings", []), "n_searched": len(papers),
            "papers": papers}


def format_report(results: List[dict], limit: int = 3) -> str:
    out = ["LITERATURE FOR THIS ANALYSIS (Valency MCP Bond)", "=" * 48]
    for r in results:
        hits = [p for p in r["papers"] if p["mentions_phase"]]
        out.append(f"\n{r['label']}   [Valency confidence: {r['confidence'] or 'n/a'}]")
        out.append(f"  query: {r['query']}")
        if r["label"].startswith("unknown peak"):
            out.append(f"  {len(hits)} candidate papers (no name check for unknown peaks)")
        else:
            out.append(f"  {len(hits)} of {r['n_searched']} papers found mention "
                       f"{_mineral_stem(r['label'])} in title or abstract")
        if not hits:
            out.append("  -> NO direct literature found: nothing is cited for this phase.")
            continue
        for i, p in enumerate(hits[:limit], 1):
            a = p["authors"]
            who = (a[0] + (" et al" if len(a) > 1 else "")) if a else "unknown authors"
            link = f"https://doi.org/{p['doi']}" if p["doi"] else p["url"]
            out.append(f"  {i}. {p['title']} ({p['year']}). {who}. {link}")
    return "\n".join(out)


def literature_search(phases: List[str], unknown_peaks_cm: List[float], token: str,
                      limit: int = 3, topic: str = "raman", endpoint: str = ENDPOINT,
                      client: Optional["ValencyClient"] = None) -> List[dict]:
    """Structured results (one dict per phase/peak), ready for a report or an LLM."""
    client = client or ValencyClient(token, endpoint)
    client.connect()
    tools = {t["name"]: t for t in client.list_tools()}
    if SEARCH_TOOL not in tools:
        raise MCPError(f"'{SEARCH_TOOL}' not offered by the server; tools: {sorted(tools)[:10]}")
    n_fetch = min(max(limit * 3, 8), 10)     # fetch extra, keep the on-topic ones
    results = []
    for label, query in build_queries(phases, unknown_peaks_cm, topic).items():
        res = client.call(SEARCH_TOOL, _query_args(tools[SEARCH_TOOL], query, n_fetch))
        r = check_papers(label, _parse_search(_content_text(res)))
        r["query"] = query
        results.append(r)
    return results


def literature_report(phases: List[str], unknown_peaks_cm: List[float], token: str,
                      limit: int = 3, endpoint: str = ENDPOINT, topic: str = "raman") -> str:
    return format_report(literature_search(phases, unknown_peaks_cm, token, limit, topic, endpoint), limit)


def list_server_tools(token: str, endpoint: str = ENDPOINT) -> str:
    client = ValencyClient(token, endpoint)
    client.connect()
    lines = []
    for t in client.list_tools():
        props = (t.get("inputSchema") or {}).get("properties", {})
        args = ", ".join(f"{k}:{v.get('type', '?')}" + (f"={v['enum']}" if "enum" in v else "")
                         for k, v in props.items())
        lines.append(f"- {t['name']}({args})\n    {(t.get('description') or '').strip()[:200]}")
    return "\n".join(lines)


def main():
    try:
        sys.stdout.reconfigure(errors="replace")     # Windows consoles / redirected output
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description="Find papers for identified phases with Valency MCP Bond")
    ap.add_argument("phases", nargs="*", help="identified minerals, e.g. Grossular Quartz")
    ap.add_argument("--unknown", nargs="*", type=float, default=[], help="unexplained Raman peaks (cm-1)")
    ap.add_argument("--limit", type=int, default=3, help="papers shown per phase")
    ap.add_argument("--topic", choices=["raman", "xrd"], default="raman",
                    help="raman: band assignment; xrd: crystal structure / unit cell")
    ap.add_argument("--json", action="store_true", help="print structured results as JSON")
    ap.add_argument("--tools", action="store_true", help="list the server's tools and their arguments")
    ap.add_argument("--endpoint", default=ENDPOINT)
    a = ap.parse_args()
    token = os.environ.get("VALENCY_TOKEN", "")
    try:
        if a.tools:
            print(list_server_tools(token, a.endpoint))
        elif a.json:
            print(json.dumps(literature_search(a.phases, a.unknown, token, a.limit, a.topic, a.endpoint),
                             indent=1, ensure_ascii=False))
        else:
            print(literature_report(a.phases, a.unknown, token, a.limit, a.endpoint, a.topic))
    except MCPError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
