"""RL01: a lake read resolves the selected feature order, causal timestamps
and immutable resource identities; no reserved future data enter an
observation."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ._fixtures import (FEATURES, WINDOW, env_config, make_env, sha256_file,
                        write_manifest, write_synthetic_csv)
from ._mechanisms import assert_checkout_resolution, require

pytest.importorskip("gymnasium")
pytest.importorskip("backtrader")


@pytest.fixture
def lake(tmp_path):
    csv = write_synthetic_csv(tmp_path / "lake" / "fixture.csv")
    manifest = write_manifest(tmp_path / "lake" / "manifest.json", csv, status="FROZEN")
    return {"csv": csv, "manifest": manifest, "dir": tmp_path / "lake"}


def test_binding_resolves_feature_order_and_resource_identity(lake):
    assert_checkout_resolution("RL01")
    Binding = require("RL01", "rl_temporal.lake_binding", "SelectedFeatureBinding",
                      "selected-feature manifest binding (feature order + resource sha)")
    binding = Binding.from_manifest(lake["manifest"])
    assert binding.feature_order == tuple(FEATURES)
    assert binding.manifest_sha256 == sha256_file(lake["manifest"])
    assert binding.status == "FROZEN"
    resources = binding.verify_resources()
    assert resources["model_ready_view"]["sha256"] == sha256_file(lake["csv"])
    assert resources["model_ready_view"]["verified"] is True
    overrides = binding.env_overrides()
    assert overrides["feature_columns"] == list(FEATURES)
    assert overrides["selected_feature_manifest_sha256"] == binding.manifest_sha256


def test_binding_refuses_a_tampered_resource(lake):
    Binding = require("RL01", "rl_temporal.lake_binding", "SelectedFeatureBinding",
                      "immutable resource identity check")
    lake["csv"].write_text(lake["csv"].read_text() + "2099-01-01 00:00:00,1,1,1,1,1,0,0,0\n")
    binding = Binding.from_manifest(lake["manifest"])
    with pytest.raises(Exception, match="sha256"):
        binding.verify_resources()


def test_draft_manifest_refuses_a_real_data_pilot(lake):
    Binding = require("RL01", "rl_temporal.lake_binding", "SelectedFeatureBinding",
                      "DRAFT_NOT_FROZEN refusal of real-data pilots")
    PilotRefused = require("RL01", "rl_temporal.lake_binding", "PilotRefused",
                           "typed pilot refusal")
    draft = write_manifest(lake["dir"] / "draft.json", lake["csv"], status="DRAFT_NOT_FROZEN")
    binding = Binding.from_manifest(draft)
    assert binding.status == "DRAFT_NOT_FROZEN"
    with pytest.raises(PilotRefused, match="DRAFT_NOT_FROZEN"):
        binding.refuse_real_data_pilot()
    Binding.from_manifest(lake["manifest"]).refuse_real_data_pilot()  # FROZEN passes


def test_observation_at_step_t_ignores_rows_at_or_after_t(lake):
    probe = require("RL01", "rl_temporal.lake_binding", "causal_timestamp_probe",
                    "causal timestamp probe (future rows cannot move an observation)")
    cfg = env_config(lake["csv"])
    report = probe(make_env, cfg, step=120)
    assert report["future_rows_influence"] is False, report
    assert report["past_rows_influence"] is True, report
    assert report["observation_rows"] == [120 - WINDOW, 120]


def test_manifest_without_selected_variant_is_refused(lake):
    Binding = require("RL01", "rl_temporal.lake_binding", "SelectedFeatureBinding",
                      "explicit variant selection")
    doc = json.loads(lake["manifest"].read_text())
    doc.pop("selected_variant")
    path = lake["dir"] / "novariant.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(Exception, match="selected_variant"):
        Binding.from_manifest(path)
