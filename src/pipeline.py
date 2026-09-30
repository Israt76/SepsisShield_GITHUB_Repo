"""End-to-end inference: raw hourly measurements -> risk, uncertainty, trust, explanation."""
from __future__ import annotations
import json, pickle
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from features import make_features
import integrity

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


@lru_cache(maxsize=1)
def load():
    cfg = json.load(open(MODELS / "config.json"))
    models = [lgb.Booster(model_file=str(MODELS / f"lgb_seed{s}.txt")) for s in cfg["seeds"]]
    iso = pickle.load(open(MODELS / "calibrator.pkl", "rb"))
    icfg = integrity.load_cfg()
    return cfg, models, iso, icfg


def score(raw: pd.DataFrame) -> pd.DataFrame:
    """raw: hourly rows (one or many patients) with the original PhysioNet columns + patient_id, hour."""
    cfg, models, iso, icfg = load()
    raw = raw.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    if "hospital" not in raw:
        raw["hospital"] = "NA"
    feats = make_features(raw)
    X = feats[cfg["features"]]
    P = np.column_stack([m.predict(X) for m in models])
    out = raw[["patient_id", "hour"]].copy()
    if "SepsisLabel" in raw:
        out["SepsisLabel"] = raw["SepsisLabel"].to_numpy()
    out["prob_raw"] = P.mean(1)
    Pc = np.clip(P, 1e-6, 1 - 1e-6)
    out["ens_std"] = np.log(Pc / (1 - Pc)).std(1)       # disagreement on log-odds scale
    out["ens_std_prob"] = P.std(1)                       # same, in probability points (for display)
    out["risk"] = iso.predict(out["prob_raw"].to_numpy())
    out["alert"] = out["risk"] >= cfg["threshold"]
    chk = integrity.check(raw, icfg)
    out = pd.concat([out, chk.reset_index(drop=True)], axis=1)
    out["trust"] = integrity.trust_level(out, out["ens_std"].to_numpy(), icfg["ens_std_thr"])
    # trust-aware abstention policy (research prototype)
    out["decision"] = np.select([out["trust"] == "LOW", out["trust"] == "REDUCED"],
                                ["WITHHELD", "SHOW_WITH_WARNING"], "SHOW")
    out["alert_shown"] = out["alert"] & (out["decision"] != "WITHHELD")
    return out, feats


def explain(feats_row: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """SHAP contributions (log-odds, averaged over the ensemble) for one hour."""
    cfg, models, _, _ = load()
    X = feats_row[cfg["features"]]
    contrib = np.mean([m.predict(X, pred_contrib=True) for m in models], axis=0)[0]
    vals = pd.Series(contrib[:-1], index=cfg["features"])
    order = vals.abs().sort_values(ascending=False).index[:top]
    return pd.DataFrame({"feature": order, "value": X.iloc[0][order].to_numpy(),
                         "shap": vals[order].to_numpy()})
