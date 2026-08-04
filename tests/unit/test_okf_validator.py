"""K1 validator tests: stale, contradictory, malformed, secret-bearing and
missing-source concepts must fail closed; the real bundle must be clean."""
import importlib.util
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "okf_validate", REPO_ROOT / "tools" / "okf_validate.py"
)
okf = importlib.util.module_from_spec(_SPEC)
sys.modules["okf_validate"] = okf
_SPEC.loader.exec_module(okf)

AS_OF = date(2026, 8, 4)


def _concept(**overrides):
    fields = {
        "type": "concept",
        "id": "sample-concept",
        "title": "Sample",
        "status": "draft",
        "producer": "satoshi-iii",
        "verified_by": "none",
        "created": "2026-08-04",
        "updated": "2026-08-04",
        "review_by": "2026-09-04",
        "canonical_for": "sample-topic",
        "supersedes": "none",
    }
    fields.update({k: v for k, v in overrides.items()
                   if k not in ("sources", "body", "extra")})
    sources = overrides.get("sources", ["docs/work_plan/README.md"])
    body = overrides.get("body", "A factual body citing its sources.")
    lines = ["---"]
    for key, value in fields.items():
        lines.append(f"{key}: {value}")
    lines.append("sources:")
    for source in sources:
        lines.append(f"  - {source}")
    lines.append("tags:")
    lines.append("  - test")
    if overrides.get("extra"):
        lines.append(overrides["extra"])
    lines.append("---")
    lines.append(body)
    return "\n".join(lines) + "\n"


def _bundle(tmp_path, files):
    bundle = tmp_path / "okf"
    bundle.mkdir()
    (tmp_path / "docs" / "work_plan").mkdir(parents=True)
    (tmp_path / "docs" / "work_plan" / "README.md").write_text("canon\n")
    for name, text in files.items():
        (bundle / name).write_text(text)
    return bundle


def _errors(tmp_path, files, as_of=AS_OF):
    bundle = _bundle(tmp_path, files)
    errors, _ = okf.validate_bundle(bundle, tmp_path, as_of)
    return errors


def test_real_bundle_is_clean_and_manifest_reproducible():
    errors, files = okf.validate_bundle(
        REPO_ROOT / "knowledge" / "okf", REPO_ROOT, AS_OF)
    assert errors == []
    assert len(files) == 8
    manifest = okf.compute_manifest(REPO_ROOT / "knowledge" / "okf")
    on_disk = (REPO_ROOT / "knowledge" / "okf" / "MANIFEST.sha256").read_text()
    assert manifest == on_disk                     # reproducible byte-stable
    assert manifest == okf.compute_manifest(REPO_ROOT / "knowledge" / "okf")


def test_valid_synthetic_concept_passes(tmp_path):
    errors = _errors(tmp_path, {
        "sample-concept.md": _concept(),
    })
    assert errors == []


def test_stale_review_by_fails(tmp_path):
    errors = _errors(tmp_path, {
        "sample-concept.md": _concept(review_by="2026-08-20"),
    }, as_of=date(2026, 9, 1))
    assert any("STALE" in e for e in errors)


def test_malformed_frontmatter_fails(tmp_path):
    errors = _errors(tmp_path, {
        "sample-concept.md": "---\ntype concept broken\n---\nbody\n",
    })
    assert any("unparseable" in e for e in errors)


def test_unknown_key_fails(tmp_path):
    errors = _errors(tmp_path, {
        "sample-concept.md": _concept(extra="surprise: value"),
    })
    assert any("unknown keys" in e for e in errors)


def test_duplicate_id_fails(tmp_path):
    errors = _errors(tmp_path, {
        "sample-concept.md": _concept(),
        "other-name.md": _concept(),               # same id, wrong stem too
    })
    assert any("duplicate id" in e for e in errors)


def test_contradiction_without_supersession_fails(tmp_path):
    errors = _errors(tmp_path, {
        "first-view.md": _concept(id="first-view"),
        "second-view.md": _concept(id="second-view"),
    })
    assert any("CONTRADICTION" in e for e in errors)


def test_supersession_resolves_contradiction(tmp_path):
    errors = _errors(tmp_path, {
        "first-view.md": _concept(id="first-view"),
        "second-view.md": _concept(id="second-view", supersedes="first-view"),
    })
    assert errors == []


def test_secret_and_account_patterns_fail(tmp_path):
    errors = _errors(tmp_path, {
        "sample-concept.md": _concept(
            body="The account DU1234567 uses token: abcd1234efgh"),
    })
    assert any("prohibited" in e for e in errors)


def test_missing_source_fails(tmp_path):
    errors = _errors(tmp_path, {
        "sample-concept.md": _concept(sources=["docs/absent.md"]),
    })
    assert any("missing source" in e for e in errors)


def test_source_escape_fails(tmp_path):
    errors = _errors(tmp_path, {
        "sample-concept.md": _concept(sources=["../outside.md"]),
    })
    assert any("escapes the repository" in e for e in errors)


def test_manifest_tamper_detected(tmp_path):
    bundle = _bundle(tmp_path, {"sample-concept.md": _concept()})
    (bundle / "MANIFEST.sha256").write_text(okf.compute_manifest(bundle))
    (bundle / "sample-concept.md").write_text(_concept(title="Altered"))
    assert (bundle / "MANIFEST.sha256").read_text() != okf.compute_manifest(bundle)
