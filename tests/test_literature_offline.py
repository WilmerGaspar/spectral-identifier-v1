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
        schema = {"properties": {"query": {"type": "string"}, "limit": {"type": "integer"},
                                 "include_abstract": {"type": "boolean"}}}
        return [{"name": L.SEARCH_TOOL, "inputSchema": schema},
                {"name": L.KEYWORD_TOOL, "inputSchema": schema}]

    def call(self, name, args):
        assert args["include_abstract"] is True
        if name == L.KEYWORD_TOOL:
            papers = [{"id": "k1", "title": "Synchrotron study of yttria-stabilized zirconia",
                       "abstract": "", "authors": ["Z"]}]
            if args["query"].startswith("vesuvianite"):
                papers.append({"id": "k2", "title": "The crystal structure of Ti-rich vesuvianite",
                               "abstract": "", "authors": ["V"]})
            return {"content": [{"type": "text", "text": json.dumps({"papers": papers})}]}
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
    flags = [p["mentions_phase"] for p in papers]
    assert flags[0] and sum(flags) == 1            # on-topic paper first, look-alikes dropped
    report = L.format_report(res)
    assert f"1 of {len(papers)} papers" in report and "sidorenkite" not in report.lower()


def test_no_hits_means_nothing_cited():
    res = L.literature_search(["Quartz"], [], "token", client=FakeClient())
    assert "NO direct literature found" in L.format_report(res)


def test_keyword_search_finds_rare_mineral():
    res = L.literature_search(["Vesuvianite"], [], "token", client=FakeClient())
    hits = [p for p in res[0]["papers"] if p["mentions_phase"]]
    assert len(hits) == 1 and hits[0]["found_by"] == "keyword"


def test_zirconia_is_not_zircon():
    res = L.literature_search(["Zircon"], [], "token", client=FakeClient())
    assert not any(p["mentions_phase"] for p in res[0]["papers"])


def test_mineral_stem():
    assert L._mineral_stem("Xenotime-(Y)") == "xenotime"


if __name__ == "__main__":
    test_off_target_papers_are_dropped()
    test_no_hits_means_nothing_cited()
    test_keyword_search_finds_rare_mineral()
    test_zirconia_is_not_zircon()
    test_mineral_stem()
    print("ok")
