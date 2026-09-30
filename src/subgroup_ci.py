"""Subgroup audit with patient counts, sepsis cases and patient-bootstrap 95% CIs (AUROC, patient sensitivity)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from metrics import onset_hour, patient_level

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main(B=300, seed=0):
    thr = json.load(open(ROOT / "models" / "config.json"))["threshold"]
    d = pd.read_parquet(RES / "test_predictions.parquet")
    d["age_group"] = pd.cut(d["Age"], [0, 45, 65, 80, 200], labels=["<45", "45-64", "65-79", "80+"], right=False).astype(str)
    d["sex"] = d["Gender"].map({0: "Female", 1: "Male"})
    d["icu"] = np.select([d["Unit1"] == 1, d["Unit2"] == 1], ["MICU", "SICU"], "Unknown")
    d["pred"] = (d["prob"] >= thr).astype(int)
    ts = onset_hour(d); d["t_s"] = d["patient_id"].map(ts)
    win = d["t_s"].notna() & (d["hour"] >= d["t_s"] - 12) & (d["hour"] <= d["t_s"] + 3)
    det = d[win].groupby("patient_id")["pred"].max()
    rng = np.random.default_rng(seed)
    rows = []
    for col in ("hospital", "age_group", "sex", "icu"):
        for val, g in d.groupby(col, observed=True):
            pats = g["patient_id"].unique()
            sep = [p for p in pats if p in det.index or not np.isnan(ts.get(p, np.nan))]
            sep = [p for p in pats if not np.isnan(ts.get(p, np.nan))]
            idx = g.groupby("patient_id").indices
            y, p = g["SepsisLabel"].to_numpy(), g["prob"].to_numpy()
            au, se = [], []
            sep_arr = np.array(sep, dtype=object)
            detv = det.reindex(sep_arr).fillna(0).to_numpy()
            for _ in range(B):
                s = rng.choice(pats, len(pats))
                ii = np.concatenate([idx[q] for q in s])
                if y[ii].min() != y[ii].max():
                    au.append(roc_auc_score(y[ii], p[ii]))
                if len(sep_arr):
                    se.append(detv[rng.integers(0, len(sep_arr), len(sep_arr))].mean())
            pl = patient_level(g, g["pred"].to_numpy())
            rows.append({"attribute": col, "group": str(val), "n_patients": int(len(pats)), "n_septic": int(len(sep)),
                         "auroc": float(roc_auc_score(y, p)), "auroc_ci": [float(np.percentile(au, 2.5)), float(np.percentile(au, 97.5))],
                         "patient_sensitivity": pl["patient_sensitivity"],
                         "sens_ci": [float(np.percentile(se, 2.5)), float(np.percentile(se, 97.5))],
                         "patient_specificity": pl["patient_specificity"],
                         "mean_pred": float(g["prob"].mean()), "observed_rate": float(y.mean())})
    json.dump(rows, open(RES / "subgroups_ci.json", "w"), indent=1)
    print(pd.DataFrame(rows).drop(columns=["mean_pred"]).round(3).to_string())


if __name__ == "__main__":
    main()
