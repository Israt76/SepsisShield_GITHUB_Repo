"""Evaluation metrics, including the official PhysioNet 2019 normalized utility score."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1.0, -2.0, -0.05, 0.0


def onset_hour(df: pd.DataFrame) -> pd.Series:
    """t_sepsis per patient (challenge convention: labels start 6h before onset).
    NaN for non-septic patients."""
    first = df.loc[df["SepsisLabel"] == 1].groupby("patient_id")["hour"].min()
    return (first - DT_OPTIMAL).reindex(df["patient_id"].unique())


def _row_utility(tt: np.ndarray, septic: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Vectorised re-implementation of compute_prediction_utility (evaluate_sepsis_score.py)."""
    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY); b1 = -m1 * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL); b2 = -m2 * DT_LATE
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL); b3 = -m3 * DT_OPTIMAL
    u = np.zeros(len(tt))
    # non-septic
    ns = ~septic
    u[ns & (pred == 1)] = U_FP
    u[ns & (pred == 0)] = U_TN
    s = septic & (tt <= DT_LATE)
    early = s & (tt <= DT_OPTIMAL)
    late = s & (tt > DT_OPTIMAL)
    p1, p0 = pred == 1, pred == 0
    u[early & p1] = np.maximum(m1 * tt[early & p1] + b1, U_FP)
    u[late & p1] = m2 * tt[late & p1] + b2
    u[early & p0] = 0.0
    u[late & p0] = m3 * tt[late & p0] + b3
    return u


def utility_score(df: pd.DataFrame, pred: np.ndarray) -> float:
    """Normalized utility: (observed - inaction) / (best - inaction). df needs patient_id, hour, SepsisLabel."""
    ts = onset_hour(df)
    t_s = df["patient_id"].map(ts).to_numpy(dtype=float)
    septic = ~np.isnan(t_s)
    tt = np.where(septic, df["hour"].to_numpy() - t_s, 0)
    best = (septic & (tt >= DT_EARLY) & (tt <= DT_LATE)).astype(int)
    obs = _row_utility(tt, septic, pred.astype(int)).sum()
    b = _row_utility(tt, septic, best).sum()
    n = _row_utility(tt, septic, np.zeros_like(best)).sum()
    return float((obs - n) / (b - n))


def best_threshold(df: pd.DataFrame, prob: np.ndarray, grid=None) -> tuple[float, float]:
    grid = np.concatenate([np.linspace(0.005, 0.1, 39), np.linspace(0.11, 0.5, 40)]) if grid is None else grid
    scores = [utility_score(df, (prob >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins (robust under heavy class imbalance)."""
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def patient_level(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Alert-level clinical metrics.
    - sensitivity: septic patients with >=1 alert in [onset-12h, onset+3h]
    - specificity: non-septic patients who never receive an alert
    - lead time: onset - first alert hour, for alerts within [onset-12h, onset]
    - false alarms per 100 non-septic patient-days
    """
    d = df[["patient_id", "hour"]].copy()
    d["pred"] = pred.astype(int)
    ts = onset_hour(df)
    d["t_s"] = d["patient_id"].map(ts)
    septic_ids = ts.dropna().index
    s = d[d["patient_id"].isin(septic_ids)]
    win = s[(s["hour"] >= s["t_s"] + DT_EARLY) & (s["hour"] <= s["t_s"] + DT_LATE) & (s["pred"] == 1)]
    detected = win.groupby("patient_id")["hour"].min()
    sens = len(detected) / max(len(septic_ids), 1)
    pre = s[(s["hour"] >= s["t_s"] + DT_EARLY) & (s["hour"] <= s["t_s"]) & (s["pred"] == 1)]
    first_pre = pre.groupby("patient_id")["hour"].min()
    lead = (ts.loc[first_pre.index] - first_pre)
    ns = d[~d["patient_id"].isin(septic_ids)]
    alerted = ns.groupby("patient_id")["pred"].max()
    spec = 1 - alerted.mean() if len(alerted) else np.nan
    fa_per_100_days = ns["pred"].sum() / (len(ns) / 24) * 100 if len(ns) else np.nan
    return {
        "patient_sensitivity": float(sens),
        "patient_specificity": float(spec),
        "median_lead_time_h": float(lead.median()) if len(lead) else float("nan"),
        "pct_detected_ge6h_early": float((lead >= 6).sum() / max(len(septic_ids), 1)),
        "false_alert_hours_per_100_patient_days": float(fa_per_100_days),
        "n_septic": int(len(septic_ids)),
        "n_nonseptic": int(len(alerted)),
    }


def hourly(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "prevalence": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
    }


def full_report(df: pd.DataFrame, prob: np.ndarray, thr: float) -> dict:
    y = df["SepsisLabel"].to_numpy()
    pred = (prob >= thr).astype(int)
    r = hourly(y, prob)
    r["utility"] = utility_score(df, pred)
    r["threshold"] = thr
    r.update(patient_level(df, pred))
    return r
