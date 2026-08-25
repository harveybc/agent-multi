#!/usr/bin/env python3
"""WP1/R05 acceptance validator (v3: HEURISTIC claim-binding lint gate).

Coverage statement (SOTA-C05): this gate detects quantitative claims in
top-level bullets (keyword+digit heuristic) and in Markdown table rows
with numeric cells. It does NOT parse prose paragraphs or nested
bullets; its claim count means "claims detected by the heuristic",
never "all quantitative claims". It is a lint gate, not an exhaustive
verifier. for the SOTA review (order 2026-08-24).

Rejects: unknown source IDs, paper sections without a source line,
duplicate registry IDs, and secondary sources cited as the source of
a paper's own numbers. Every aspect file's per-paper section must
carry a line `Fuente: [ID loc:...]` naming registered IDs with
table/eq/page locators.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent


def doc_files(docs):
    return sorted(docs.glob("0[1-8]_*.md")) + sorted(docs.glob("10_*.md"))


FILES = doc_files(DOCS)
SOURCE_LINE = re.compile(r"Fuente[s]?:\s*(\[[^\]]+\](?:\s*,?\s*\[[^\]]+\])*)")
REF = re.compile(r"\[([A-Z0-9-]+)\s+loc:([^\]]+)\]")



# ---- v2 (SOTA-R05): claim-level binding -------------------------------
# A "quantitative claim bullet" is any top-level list bullet containing a
# digit plus a metric/config keyword. Every such bullet must itself carry
# at least one inline `[ID loc:...]` reference with a non-generic locator.
CLAIM_KW = re.compile(
    r"(?i)(sharpe|sortino|mae\b|rmse|mse\b|auc|r\u00b2|r2\b|drawdown|mdd|"
    r"retorno|rendimiento|precisi|accuracy|\bbp\b|\bpb\b|bps|"
    r"par[a\u00e1]metros|params|[e\u00e9]pocas|epochs|capas|layers|unidades|units|"
    r"learning.rate|\blr\b|gamma|\u03b3|semillas|seeds|coste|fee|comisi|"
    r"filas|rows|acciones|stocks|contratos|horizonte|ventana|lookback|"
    r"batch|buffer|volatilidad|apalancamiento|leverage|trades|operaciones|"
    r"turnover|alpha|percentil)")
NUM_RE = re.compile(r"\d")
LOC_GENERIC = {"paper", "general", "passim", "texto", "todo", "n/a", "-",
               "varios", "ver paper", "ver texto"}
LOC_ANCHOR = re.compile(
    r"(?i)^(tab|table|tabla|eq|ec|fig|p\b|pp|pag|sec|\u00a7|abs|abstract|"
    r"alg|app|apx|anexo|cap|kw|nota|benchmark)")


def locator_ok(loc, source_type):
    loc = loc.strip()
    if len(loc) < 3 or loc.lower() in LOC_GENERIC:
        return False
    if source_type == "internal_primary":
        return True  # repo-artifact locators are free text (paths/manifests)
    return bool(re.search(r"\d", loc) or LOC_ANCHOR.match(loc))


def iter_bullets(text):
    lines = text.split("\n")
    j = 0
    while j < len(lines):
        if re.match(r"^- ", lines[j]):
            blk = [lines[j]]
            j += 1
            while j < len(lines) and lines[j].startswith("  "):
                blk.append(lines[j])
                j += 1
            yield "\n".join(blk)
        else:
            j += 1


def iter_numeric_table_blocks(text):
    """Yield contiguous Markdown table blocks containing numeric cells."""
    lines = text.split("\n")
    j = 0
    while j < len(lines):
        if lines[j].lstrip().startswith("|"):
            blk = []
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                blk.append(lines[j])
                j += 1
            # a Fuente line under the table (blank lines allowed)
            # binds the whole table
            k = j
            while k < len(lines) and not lines[k].strip():
                k += 1
            trailing = lines[k] if k < len(lines) else ""
            body = "\n".join(blk)
            if re.search(r"\|[^|]*\d", body.replace("---", "")):
                yield body, trailing
        else:
            j += 1


def claim_level_problems(path, text, ids):
    probs = []
    for body, trailing in iter_numeric_table_blocks(text):
        if not (REF.search(body) or REF.search(trailing)):
            head = body.split("\n")[0][:60]
            probs.append(
                f"{path.name} :: tabla numerica sin fuente inline ni "
                f"linea Fuente inmediata: '{head}'")
    for blk in iter_bullets(text):
        if not (NUM_RE.search(blk) and CLAIM_KW.search(blk)):
            continue
        refs = REF.findall(blk)
        head = blk.split("\n")[0][:60]
        if not refs:
            probs.append(f"{path.name} :: claim sin fuente inline: '{head}'")
            continue
        for sid, loc in refs:
            if sid not in ids:
                probs.append(f"{path.name} :: claim con id desconocido {sid}: '{head}'")
                continue
            stype = ids[sid].get("type", "")
            if not locator_ok(loc, stype):
                probs.append(
                    f"{path.name} :: locator generico/invalido '{loc.strip()[:40]}' en claim '{head}'")
    return probs


def registry_metadata_problems(ids):
    probs = []
    for sid, e in ids.items():
        if "retrieved" not in e or "retrieval_channel" not in e:
            probs.append(f"registry :: {sid} sin retrieved/retrieval_channel")
        if e.get("retrieval_channel") == "local_pdf" and not e.get("content_sha256"):
            probs.append(f"registry :: {sid} local_pdf sin content_sha256")
    return probs


def main(docs_root=None) -> int:
    docs = Path(docs_root) if docs_root else DOCS
    here = docs / "sources"
    files = doc_files(docs)
    raw = json.loads((here / "registry.json").read_text())
    ids = raw["sources"]
    # duplicate detection must operate on the raw text (json.loads
    # silently keeps the last duplicate key)
    keys = re.findall(r'"([A-Z0-9-]+)":\s*\{"type"', (here / "registry.json").read_text())
    dups = {k for k in keys if keys.count(k) > 1}
    problems = []
    if dups:
        problems.append(f"duplicate registry ids: {sorted(dups)}")
    for path in files:
        text = path.read_text()
        sections = re.split(r"\n## ", text)
        for sec in sections[1:]:
            title = sec.split("\n", 1)[0]
            if "Tabla de contenido" in title or "name=\"nuestros\"" in title.lower():
                continue
            lines = SOURCE_LINE.findall(sec)
            refs = [m for line in lines for m in REF.findall(line)]
            if not refs:
                problems.append(f"{path.name} :: seccion '{title[:50]}' sin linea Fuente con locator")
                continue
            for sid, loc in refs:
                if sid not in ids:
                    problems.append(f"{path.name} :: id desconocido {sid}")
                elif not loc.strip():
                    problems.append(f"{path.name} :: {sid} sin locator")
                elif ids[sid]["type"].startswith("paper_secondary") and "propio" not in loc:
                    # secondary benchmarks may only source their OWN claims
                    if "benchmark" not in sec[:400].lower() and "control negativo" not in sec.lower() and "contaminaci" not in sec.lower():
                        problems.append(f"{path.name} :: fuente secundaria {sid} citada como primaria en '{title[:40]}'")
    for path in files:
        problems += claim_level_problems(path, path.read_text(), ids)
    problems += registry_metadata_problems(ids)
    if problems:
        print(json.dumps({"outcome": "REJECTED", "problems": problems[:40],
                          "total_problems": len(problems)}, indent=1))
        return 2
    print(json.dumps({"outcome": "PASS", "coverage": "heuristic_lint",
                      "files": len(files),
                      "registered_sources": len(ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
