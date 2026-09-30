"""Causality: a feature at hour t must depend only on measurements at hours <= t."""
import json
import numpy as np
import pandas as pd
import pytest

from features import make_features, feature_columns


def _one(demo_raw, pid):
    return demo_raw[demo_raw["patient_id"] == pid].reset_index(drop=True)


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
def test_truncated_history_gives_identical_features(demo_raw, k):
    """Compute features on the full stay and on every prefix; values at shared hours must be identical."""
    pid = demo_raw["patient_id"].unique()[k]
    p = _one(demo_raw, pid)
    full = make_features(p)
    cols = feature_columns(full)
    for cut in sorted({3, len(p) // 3, len(p) // 2, len(p) - 1}):
        part = make_features(p.iloc[:cut].reset_index(drop=True))
        a = part[cols].to_numpy(float)
        b = full.iloc[:cut][cols].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), f"look-ahead detected for {pid} at cut {cut}"


def test_future_values_do_not_change_past(demo_raw):
    """Perturbing hour t+1.. must not alter any feature at hours <= t."""
    p = _one(demo_raw, demo_raw["patient_id"].unique()[0]).copy()
    t = len(p) // 2
    base = make_features(p)
    q = p.copy()
    from data import VITALS, LABS
    for c in VITALS + LABS:                       # every raw measurement
        q.loc[q["hour"] > t, c] = q.loc[q["hour"] > t, c] * 3 + 17
        q.loc[(q["hour"] > t) & q[c].isna(), c] = 50.0   # also fill future gaps (changes missingness)
    pert = make_features(q)
    cols = feature_columns(base)
    assert np.allclose(base.iloc[: t + 1][cols].to_numpy(float), pert.iloc[: t + 1][cols].to_numpy(float), equal_nan=True)


def test_label_not_a_feature(root):
    cfg = json.load(open(root / "models" / "config.json"))
    assert "SepsisLabel" not in cfg["features"]
    assert not any(c in cfg["features"] for c in ("patient_id", "hospital", "hour"))


def test_features_match_trained_model(demo_raw, root):
    cfg = json.load(open(root / "models" / "config.json"))
    f = make_features(_one(demo_raw, demo_raw["patient_id"].unique()[0]))
    assert feature_columns(f) == cfg["features"]


def test_labels_do_not_influence_features(demo_raw):
    p = _one(demo_raw, demo_raw["patient_id"].unique()[1]).copy()
    a = make_features(p)
    q = p.copy(); q["SepsisLabel"] = 1 - q["SepsisLabel"]
    b = make_features(q)
    cols = feature_columns(a)
    assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_integrity_layer_is_causal(demo_raw, k):
    """Trust at hour t must not depend on measurements after t."""
    import integrity
    p = _one(demo_raw, demo_raw["patient_id"].unique()[k])
    full = integrity.check(p)
    t = len(p) // 2
    part = integrity.check(p.iloc[: t + 1].reset_index(drop=True))
    cols = [c for c in full.columns if c.startswith("flag_")] + ["shift_score", "hours_since_flag", "input_alert"]
    a = part[cols].astype(float).to_numpy(); b = full.iloc[: t + 1][cols].astype(float).to_numpy()
    assert np.allclose(a, b, equal_nan=True)
