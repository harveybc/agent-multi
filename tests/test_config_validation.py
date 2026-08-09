"""Materialization-seam tests for the shared config validators (P2).

The frozen corpus in docs/audits/evidence/config_corpus is the fixture
set; per the disposition it must NEVER be rewritten to make the doctor
pass. The expected flags below are the audited defect classes plus the
disclosed co-resident defects (a historical document may carry more
true defects than the finding it was frozen for).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import config_validation as cv

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "docs/audits/evidence/config_corpus"

IMPLEMENTED = frozenset(
    {"lexicographic_weekly_v1", "risk_adjusted_return", "total_return"}
)

# Expected BLOCK rule ids per corpus document. bad_126 carries the
# dormant-year defect too: at 108f78d4 both classes coexisted (the year
# pop only arrived with the 142 correction) — disclosed, not suppressed.
EXPECTED = {
    "bad_108_110_113_eth_en_v1__a3422da3.json": {
        "metric_consistency",
        "asset_namespace",
        "genome_choice_repair",
    },
    "bad_126_unpinned_resolved_base__108f78d4.json": {
        "pinned_references",
        "dormant_year_fields",
    },
    "bad_142_dormant_year_fields__b0ea817b.json": {"dormant_year_fields"},
    "clean_eth_en_v2__HEAD.json": set(),
    "clean_eth_n_v2__HEAD.json": set(),
    "clean_resolved_base__HEAD.json": set(),
}


def _load(name: str) -> dict:
    return json.loads((CORPUS / name).read_text())


class TestFrozenCorpusConfusionMatrix:
    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_expected_blocking_rules_exactly(self, name):
        report = cv.evaluate(_load(name), implemented_metrics=IMPLEMENTED)
        assert set(report["blocking"]) == EXPECTED[name], (
            f"{name}: got {sorted(report['blocking'])}, "
            f"expected {sorted(EXPECTED[name])}"
        )

    @pytest.mark.parametrize(
        "name", [n for n, flags in EXPECTED.items() if flags]
    )
    def test_defective_documents_block_overall(self, name):
        report = cv.evaluate(_load(name), implemented_metrics=IMPLEMENTED)
        assert report["overall"] == cv.BLOCK

    @pytest.mark.parametrize(
        "name", [n for n, flags in EXPECTED.items() if not flags]
    )
    def test_clean_documents_pass_overall(self, name):
        report = cv.evaluate(_load(name), implemented_metrics=IMPLEMENTED)
        assert report["overall"] == cv.PASS

    def test_corpus_fixtures_match_their_frozen_hashes(self):
        import hashlib

        manifest = json.loads((CORPUS / "CORPUS_MANIFEST.json").read_text())
        for name, meta in manifest["files"].items():
            actual = hashlib.sha256((CORPUS / name).read_bytes()).hexdigest()
            assert actual == meta["sha256"], (
                f"corpus fixture {name} was rewritten after freezing"
            )

    def test_the_mislabeled_v1_at_head_is_caught(self):
        """The v1 example at HEAD is byte-identical to the defective
        fixture; an agent once labeled it clean. The doctor must not."""
        path = (
            REPO
            / "examples/config/phase_2_eth_curriculum/optimization/phase_2_eth_en_v1.json"
        )
        report = cv.evaluate(
            json.loads(path.read_text()), implemented_metrics=IMPLEMENTED
        )
        assert report["overall"] == cv.BLOCK
        assert "metric_consistency" in report["blocking"]


class TestTypedOutcomes:
    def test_unavailable_metrics_is_required_and_refuses(self):
        report = cv.evaluate(
            _load("clean_eth_en_v2__HEAD.json"), implemented_metrics=None
        )
        assert report["overall"] == cv.UNAVAILABLE
        assert "metric_resolvable" in report["unavailable_required"]
        with pytest.raises(cv.ConfigPreflightError):
            cv.preflight_or_raise(
                _load("clean_eth_en_v2__HEAD.json"),
                implemented_metrics=None,
                context="test",
            )

    def test_unimplemented_metric_blocks(self):
        report = cv.evaluate(
            {"training": {"selection_metric": "made_up_metric_v9"}},
            implemented_metrics=IMPLEMENTED,
        )
        assert "metric_resolvable" in report["blocking"]

    def test_warning_alone_does_not_refuse(self):
        # no asset token anywhere -> asset_namespace WARNING, nothing else
        report = cv.evaluate({"experiment": {"name": "x"}},
                             implemented_metrics=IMPLEMENTED)
        assert report["overall"] == cv.WARNING
        cv.preflight_or_raise(
            {"experiment": {"name": "x"}},
            implemented_metrics=IMPLEMENTED,
            context="test",
        )  # must not raise

    def test_asset_token_matching_is_delimited(self):
        assert cv._asset_tokens_in("method_together") == set()
        assert cv._asset_tokens_in("eth_curriculum_v2") == {"eth"}
        assert cv._asset_tokens_in("ethusdt_4h.csv") == {"ethusdt"}
        assert cv._asset_tokens_in("ETHUSD") == {"ethusd"}

    def test_split_overlap_blocks(self):
        report = cv.evaluate(
            {
                "data": {
                    "train_start": "2020-01-01T00:00:00",
                    "train_end": "2024-06-01T00:00:00",
                    "validation_start": "2024-01-01T00:00:00",
                    "validation_end": "2024-12-31T23:59:59",
                }
            },
            implemented_metrics=IMPLEMENTED,
        )
        assert "split_overlap" in report["blocking"]


class TestLaunchSeam:
    """Socket-free proof that the supervisor's launch path refuses a
    blocked config through the SAME shared validators."""

    def _node_config(self, tmp_path, canonical_doc):
        canonical = tmp_path / "canonical.json"
        canonical.write_text(json.dumps(canonical_doc))
        overlay = tmp_path / "overlay.json"
        overlay.write_text(json.dumps({"roots": {}}))
        return {
            "domains": [
                {
                    "optimization_config": {
                        "agent_multi_root": str(tmp_path),
                        "load_config": str(canonical),
                        "runtime_overlay": str(overlay),
                    }
                }
            ]
        }

    def test_supervisor_preflight_refuses_blocked_config(self, tmp_path):
        from app.campaign_supervisor import CampaignSupervisor

        defective = _load("bad_108_110_113_eth_en_v1__a3422da3.json")
        fake_self = SimpleNamespace(_dataset_validation_cache={})
        with pytest.raises(cv.ConfigPreflightError):
            CampaignSupervisor._validate_dataset_evidence(
                fake_self, self._node_config(tmp_path, defective)
            )

    def test_supervisor_preflight_is_the_shared_module(self):
        import inspect

        from app import campaign_supervisor

        source = inspect.getsource(
            campaign_supervisor.CampaignSupervisor._validate_dataset_evidence
        )
        assert "config_validation.preflight_or_raise" in source


class TestMetricSurfaceOwnership:
    def test_owner_module_declares_every_branch(self):
        pipeline = pytest.importorskip(
            "pipeline_plugins.rl_pipeline_with_validation"
        )
        import inspect

        surface = pipeline.IMPLEMENTED_SELECTION_METRICS
        assert "lexicographic_weekly_v1" in surface
        source = inspect.getsource(pipeline._selection_value)
        for metric in surface - {"lexicographic_weekly_v1", "total_return",
                          "paired_generalization_weekly_v1"}:
            assert f'"{metric}"' in source, (
                f"declared metric {metric!r} has no branch in _selection_value"
            )

    def test_runtime_discovery_reads_the_owner(self):
        observed = cv.runtime_implemented_metrics()
        if observed is None:
            pytest.skip("pipeline not importable in this environment")
        assert "lexicographic_weekly_v1" in observed


class TestRuffDiscoveredDefect:
    def test_execution_curriculum_aggregation_names_resolve(self):
        """Regression for the F821 latent NameError ruff's bounded
        baseline exposed (P3): `fmean` was used in the robust-scenario
        aggregation without ever being imported, so the execution-
        curriculum pipeline would have crashed at aggregation time."""
        module = pytest.importorskip(
            "pipeline_plugins.rl_pipeline_with_execution_curriculum"
        )
        assert callable(module.fmean)
        assert module.fmean([1.0, 2.0, 3.0]) == 2.0
