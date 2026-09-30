"""Integrity-layer behaviour on synthetic patients, the official utility score, and an end-to-end smoke test."""
import numpy as np
import pandas as pd
import pytest

import integrity
from data import VITALS, LABS
from metrics import utility_score


def synthetic(n=30, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: np.nan for c in VITALS + LABS}, index=range(n))
    d["HR"] = 85 + rng.normal(0, 2, n); d["SBP"] = 120 + rng.normal(0, 3, n); d["DBP"] = 65 + rng.normal(0, 2, n)
    d["MAP"] = (d["SBP"] + 2 * d["DBP"]) / 3; d["Resp"] = 17 + rng.normal(0, 1, n); d["O2Sat"] = 97 + rng.normal(0, 0.5, n)
    d["Temp"] = np.where(np.arange(n) % 4 == 0, 37 + rng.normal(0, 0.1, n), np.nan)
    d["patient_id"], d["hour"], d["hospital"] = "synthetic", np.arange(n), "A"
    for c, v in {"Age": 60.0, "Gender": 1.0, "Unit1": 1.0, "Unit2": 0.0, "HospAdmTime": -1.0}.items():
        d[c] = v
    d["ICULOS"] = np.arange(1, n + 1); d["SepsisLabel"] = 0
    return d


def test_clean_synthetic_patient_is_never_low():
    chk = integrity.check(synthetic())
    tl = integrity.trust_level(chk, np.zeros(len(chk)), 1.0)
    assert (tl != "LOW").all()


def test_fahrenheit_temperature_is_implausible_and_low():
    d = synthetic(); d.loc[20, "Temp"] = 37.0 * 1.8 + 32
    chk = integrity.check(d)
    assert chk.loc[20, "flag_implausible"] and "Temp" in chk.loc[20, "implausible_vars"]
    assert integrity.trust_level(chk, np.zeros(len(d)), 1.0)[20] == "LOW"


def test_impossible_blood_pressure_is_inconsistent():
    d = synthetic(); d.loc[15, "DBP"] = d.loc[15, "SBP"] + 5
    assert integrity.check(d).loc[15, "flag_inconsistent"]


def test_coordinated_normalising_shift_is_detected():
    d = synthetic(40)
    d.loc[:24, "HR"] += 30; d.loc[:24, "Resp"] += 8; d.loc[:24, "SBP"] -= 25; d.loc[:24, "MAP"] -= 17
    chk = integrity.check(d)          # at hour 25 the patient suddenly "recovers" on every vital at once
    assert chk.loc[25:26, "flag_coordinated_shift"].any()


def test_frozen_feed_is_detected():
    d = synthetic(30); d.loc[10:22, ["HR", "SBP", "Resp"]] = d.loc[10, ["HR", "SBP", "Resp"]].to_numpy()
    assert integrity.check(d).loc[17:22, "flag_flatline"].all()


def test_single_jump_reduces_but_does_not_veto():
    d = synthetic(); d.loc[18, "HR"] = 175
    chk = integrity.check(d)
    tl = integrity.trust_level(chk, np.zeros(len(d)), 1.0)
    assert chk.loc[18, "flag_jump"] and tl[18] == "REDUCED"


def _labels(n=40, onset_label_hour=20):
    d = pd.DataFrame({"patient_id": "s", "hour": np.arange(n)})
    d["SepsisLabel"] = (d["hour"] >= onset_label_hour).astype(int)
    ns = pd.DataFrame({"patient_id": "n", "hour": np.arange(n), "SepsisLabel": 0})
    return pd.concat([d, ns], ignore_index=True)


def test_utility_reference_points():
    d = _labels()
    t_s = 20 + 6
    best = ((d["patient_id"] == "s") & (d["hour"] >= t_s - 12) & (d["hour"] <= t_s + 3)).astype(int).to_numpy()
    assert utility_score(d, best) == pytest.approx(1.0)
    assert utility_score(d, np.zeros(len(d))) == pytest.approx(0.0)
    false_only = (d["patient_id"] == "n").astype(int).to_numpy()   # alerts only on the non-septic patient
    assert utility_score(d, false_only) < 0
    late = ((d["patient_id"] == "s") & (d["hour"] >= t_s + 1)).astype(int).to_numpy()
    assert 0 < utility_score(d, late) < utility_score(d, best)       # late alerts earn less than timely ones


def test_pipeline_smoke(demo_raw):
    import pipeline
    pid = demo_raw["patient_id"].iloc[0]
    out, feats = pipeline.score(demo_raw[demo_raw["patient_id"] == pid])
    assert out["risk"].between(0, 1).all()
    assert set(out["trust"]) <= {"HIGH", "REDUCED", "LOW"}
    assert out["decision"].isin(["SHOW", "SHOW_WITH_WARNING", "WITHHELD"]).all()
    ex = pipeline.explain(feats.iloc[[len(feats) - 1]])
    assert len(ex) == 8 and np.isfinite(ex["shap"]).all()


def test_input_trust_is_independent_of_model_output():
    """Input trust must be computed from the inputs alone: identical inputs give identical input flags
    whatever the model says, and corrupting inputs changes trust even when model agreement is perfect."""
    d = synthetic()
    chk = integrity.check(d)
    assert not any(c in chk.columns for c in ("risk", "prob_raw", "alert"))
    confident = np.zeros(len(d))          # the 5 models agree perfectly
    d2 = d.copy(); d2.loc[20, "Temp"] = 37.0 * 1.8 + 32
    t_clean = integrity.trust_level(integrity.check(d), confident, 1.0)
    t_bad = integrity.trust_level(integrity.check(d2), confident, 1.0)
    assert t_clean[20] != "LOW" and t_bad[20] == "LOW"
