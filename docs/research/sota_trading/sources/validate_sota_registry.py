#!/usr/bin/env python3
"""WP1 acceptance validator for the SOTA review (order 2026-08-24).

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
FILES = sorted(DOCS.glob("0[1-8]_*.md")) + sorted(DOCS.glob("10_*.md"))
SOURCE_LINE = re.compile(r"Fuente[s]?:\s*(\[[^\]]+\](?:\s*,?\s*\[[^\]]+\])*)")
REF = re.compile(r"\[([A-Z0-9-]+)\s+loc:([^\]]+)\]")


def main() -> int:
    raw = json.loads((HERE / "registry.json").read_text())
    ids = raw["sources"]
    # duplicate detection must operate on the raw text (json.loads
    # silently keeps the last duplicate key)
    keys = re.findall(r'"([A-Z0-9-]+)":\s*\{"type"', (HERE / "registry.json").read_text())
    dups = {k for k in keys if keys.count(k) > 1}
    problems = []
    if dups:
        problems.append(f"duplicate registry ids: {sorted(dups)}")
    for path in FILES:
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
    if problems:
        print(json.dumps({"outcome": "REJECTED", "problems": problems[:40],
                          "total_problems": len(problems)}, indent=1))
        return 2
    print(json.dumps({"outcome": "PASS", "files": len(FILES),
                      "registered_sources": len(ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
