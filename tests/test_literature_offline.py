"""Offline test of the Valency literature report (no network, no token)."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ai import literature as L


class FakeClient:
    """Stands in for the Valency server: returns a fixed search result."""
    def connect(self):
        return {}

    def list_tools(self):
        return [{"name": L.SEARCH_TOOL,
                 "inputSchema": {"properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}}]

    def call(self, name, args):
        papers = [
            {"title": "Raman spectra of siderite FeCO3.", "abstract": "", "authors": ["A", "B"],
             "doi": "10.1/x", "relevance_score": 0.80},
            {"title": "Vibrational study of sidorenkite", "abstract": "a phosphate-carbonate",
             "authors": ["C"], "url": "https://example.org/2", "relevance_score": 0.85},
        ]
        return {"content": [{"type": "text",
                             "text": json.dumps({"confidence": "ambiguous_match", "papers": papers})}]}


def test_off_target_papers_are_dropped():
    res = L.literature_search(["Siderite"], [], "token", client=FakeClient())
    papers = res[0]["papers"]
    assert [p["mentions_phase"] for p in papers] == [True, False]   # on-topic first
    report = L.format_report(res)
    assert "1 of 2 papers" in report and "sidorenkite" not in report.lower()


def test_no_hits_means_nothing_cited():
    res = L.literature_search(["Vesuvianite"], [], "token", client=FakeClient())
    assert "NO direct literature found" in L.format_report(res)


def test_mineral_stem():
    assert L._mineral_stem("Xenotime-(Y)") == "xenotime"


if __name__ == "__main__":
    test_off_target_papers_are_dropped()
    test_no_hits_means_nothing_cited()
    test_mineral_stem()
    print("ok")
