"""Train / validation / test isolation, and that every selection step used validation data only."""
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from conftest import needs_full_data
from metrics import best_threshold


def test_split_is_patient_level_and_disjoint():
    from train import split_patients
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({"patient_id": np.repeat([f"p{i:05d}" for i in range(n)], 5),
                       "hospital": np.repeat(rng.choice(["A", "B"], n), 5),
                       "SepsisLabel": np.repeat((rng.random(n) < 0.07).astype(int), 5)})
    tr, va, te = split_patients(df)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == n
    assert abs(len(te) / n - 0.2) < 0.01 and abs(len(va) / n - 0.1) < 0.01


def test_shipped_val_and_test_are_disjoint(root):
    te = set(pd.read_parquet(root / "results" / "test_patients.parquet")["patient_id"])
    va = set(pd.read_parquet(root / "results" / "val_predictions.parquet", columns=["patient_id"])["patient_id"])
    tp = set(pd.read_parquet(root / "results" / "test_predictions.parquet", columns=["patient_id"])["patient_id"])
    assert not (te & va)
    assert tp == te
    assert len(te) == 8068 and len(va) == 4034


def test_demo_patients_are_held_out_test_patients(root):
    te = set(pd.read_parquet(root / "results" / "test_patients.parquet")["patient_id"])
    demo = set(pd.read_parquet(root / "app" / "demo_index.parquet")["patient_id"])
    assert demo <= te, "dashboard must only show held-out test patients"


def test_calibrator_was_fitted_on_validation_only(root):
    val = pd.read_parquet(root / "results" / "val_predictions.parquet")
    iso = pickle.load(open(root / "models" / "calibrator.pkl", "rb"))
    refit = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(val["prob_raw"], val["SepsisLabel"])
    grid = np.linspace(0, 0.6, 400)
    assert np.allclose(iso.predict(grid), refit.predict(grid))


def test_threshold_was_selected_on_validation_only(root):
    val = pd.read_parquet(root / "results" / "val_predictions.parquet")
    cfg = json.load(open(root / "models" / "config.json"))
    thr, _ = best_threshold(val, val["prob"].to_numpy())
    assert abs(thr - cfg["threshold"]) < 1e-9


def test_integrity_thresholds_use_clean_training_budget(root):
    icfg = json.load(open(root / "models" / "integrity.json"))
    assert icfg["shift_clean_flag_rate"] <= 0.0026
    assert icfg["quantile"] == 0.9995


@needs_full_data
def test_full_split_reproduces_shipped_test_set(root):
    from train import split_patients
    raw = pd.read_parquet(root / "data" / "processed" / "hourly.parquet", columns=["patient_id", "hospital", "SepsisLabel"])
    tr, va, te = split_patients(raw)
    shipped = set(pd.read_parquet(root / "results" / "test_patients.parquet")["patient_id"])
    assert te == shipped
