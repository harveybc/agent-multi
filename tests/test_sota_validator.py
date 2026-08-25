"""Adversarial tests for the SOTA source validator v2 (SOTA-R05).

Each test materializes a minimal docs tree and asserts the validator's
verdict. Reproduction-before evidence: v1 passed with section-level
Fuente lines only; the fixtures below are exactly the cases v1 accepted
and v2 must refuse.
"""
import importlib.util
import json
import sys
from pathlib import Path

VALIDATOR = (Path(__file__).resolve().parents[1] /
             "docs/research/sota_trading/sources/validate_sota_registry.py")
spec = importlib.util.spec_from_file_location("sota_validator", VALIDATOR)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

GOOD_REGISTRY = {"sources": {
    "GKX2020": {"type": "paper_primary", "title": "t", "authors": "a",
                "venue": "v", "year": 2020, "url": "u",
                "retrieved": "2026-08-24",
                "retrieval_channel": "web_fulltext", "content_sha256": None},
    "OURS-X": {"type": "internal_primary", "title": "t", "authors": "a",
               "venue": "repo", "year": 2026, "url": "u",
               "retrieved": "2026-08-24",
               "retrieval_channel": "repo_artifact", "content_sha256": None},
}}

GOOD_DOC = """# 01 doc

## P1 seccion
- **Sharpe anualizado 1,35** neto de nada. Fuente: [GKX2020 loc:Tab.7]
- prosa sin numeros de resultado.

Fuentes: [GKX2020 loc:Tab.7]
"""


def build(tmp_path, doc_text, registry=None):
    docs = tmp_path / "docs"
    (docs / "sources").mkdir(parents=True)
    (docs / "01_TEST.md").write_text(doc_text)
    (docs / "sources" / "registry.json").write_text(
        json.dumps(registry or GOOD_REGISTRY))
    return docs


def run(docs, capsys):
    rc = mod.main(str(docs))
    out = capsys.readouterr().out
    return rc, json.loads(out)


def test_valid_tree_passes(tmp_path, capsys):
    rc, out = run(build(tmp_path, GOOD_DOC), capsys)
    assert rc == 0 and out["outcome"] == "PASS"


def test_quant_claim_without_inline_source_rejected(tmp_path, capsys):
    doc = GOOD_DOC.replace(
        "- **Sharpe anualizado 1,35** neto de nada. Fuente: [GKX2020 loc:Tab.7]",
        "- **Sharpe anualizado 1,35** neto de nada.")
    rc, out = run(build(tmp_path, doc), capsys)
    assert rc != 0 and any("claim sin fuente inline" in p
                           for p in out["problems"])


def test_unknown_source_id_rejected(tmp_path, capsys):
    doc = GOOD_DOC.replace("[GKX2020 loc:Tab.7]", "[NOPE99 loc:Tab.7]")
    rc, out = run(build(tmp_path, doc), capsys)
    assert rc != 0 and any("NOPE99" in p for p in out["problems"])


def test_generic_locator_rejected(tmp_path, capsys):
    doc = GOOD_DOC.replace("loc:Tab.7", "loc:paper")
    rc, out = run(build(tmp_path, doc), capsys)
    assert rc != 0 and any("locator generico" in p for p in out["problems"])


def test_internal_source_free_text_locator_allowed(tmp_path, capsys):
    doc = GOOD_DOC.replace(
        "Fuente: [GKX2020 loc:Tab.7]\n- prosa",
        "Fuente: [OURS-X loc:manifiesto verificado por ejecucion]\n- prosa")
    rc, out = run(build(tmp_path, doc), capsys)
    assert rc == 0, out


def test_registry_missing_retrieval_metadata_rejected(tmp_path, capsys):
    reg = json.loads(json.dumps(GOOD_REGISTRY))
    del reg["sources"]["GKX2020"]["retrieved"]
    rc, out = run(build(tmp_path, GOOD_DOC, reg), capsys)
    assert rc != 0 and any("sin retrieved" in p for p in out["problems"])


def test_local_pdf_without_hash_rejected(tmp_path, capsys):
    reg = json.loads(json.dumps(GOOD_REGISTRY))
    reg["sources"]["GKX2020"]["retrieval_channel"] = "local_pdf"
    rc, out = run(build(tmp_path, GOOD_DOC, reg), capsys)
    assert rc != 0 and any("local_pdf sin content_sha256" in p
                           for p in out["problems"])


def test_duplicate_registry_id_rejected(tmp_path, capsys):
    docs = build(tmp_path, GOOD_DOC)
    raw = (docs / "sources" / "registry.json").read_text()
    dup = raw.replace(
        '"GKX2020": {"type"',
        '"GKX2020": {"type": "paper_primary", "title": "shadow"}, "GKX2020": {"type"', 1)
    (docs / "sources" / "registry.json").write_text(dup)
    rc, out = run(docs, capsys)
    assert rc != 0 and any("duplicate" in p for p in out["problems"])
