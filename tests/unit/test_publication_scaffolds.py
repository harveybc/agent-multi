from pathlib import Path

from tools.validate_publication_scaffolds import PAPER_IDS, validate


def test_publication_scaffolds_match_contract() -> None:
    root = Path(__file__).resolve().parents[2] / "papers"
    assert validate(root) == []
    assert {path.name for path in root.iterdir() if path.is_dir()} == set(PAPER_IDS)
