"""Additional validation experiments.

  baselines        clinical rules (SIRS, qSOFA-partial), logistic regression, single LightGBM,
                   5-model ensemble, ensemble + isotonic calibration (= SepsisShield model)
  bootstrap_ci     patient-level bootstrap 95% CIs for AUROC, AUPRC, utility, patient sensitivity/specificity
  threshold_sweep  utility / sensitivity / specificity / alert burden across thresholds (test, and B->A transfer)
  recalibration    cross-hospital: transferred model as-is vs locally recalibrated (isotonic + threshold)
                   on 20% of destination-hospital patients, evaluated on the other 80%
All selection (thresholds, calibrators, LR scaling) uses validation / source data only.
"""
from pathlib import Path
import json, pickle, time, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from features import feature_columns
from metrics import (full_report, best_threshold, utility_score, patient_level, ece, onset_hour,
                     _row_utility, DT_EARLY, DT_LATE)
from train import split_patients

ROOT = Path(__file__).resolve().parents[1]
RES, MODELS = ROOT / "results", ROOT / "models"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def iso_fit(p, y):
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)


def summarize(d, prob, thr):
    r = full_report(d, prob, thr)
    return {k: r[k] for k in ("auroc", "auprc", "utility", "ece", "patient_sensitivity",
                              "patient_specificity", "pct_detected_ge6h_early", "threshold")}


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(d, prob, thr, B=500, seed=0):
    """Patient-level bootstrap (resample patients with replacement)."""
    d = d[["patient_id", "hour", "SepsisLabel"]].reset_index(drop=True).copy()
    d["p"] = prob
    d["pred"] = (prob >= thr).astype(int)
    ts = onset_hour(d)
    t_s = d["patient_id"].map(ts).to_numpy(float)
    septic = ~np.isnan(t_s)
    tt = np.where(septic, d["hour"].to_numpy() - t_s, 0)
    best = (septic & (tt >= DT_EARLY) & (tt <= DT_LATE)).astype(int)
    d["u_obs"] = _row_utility(tt, septic, d["pred"].to_numpy())
    d["u_best"] = _row_utility(tt, septic, best)
    d["u_none"] = _row_utility(tt, septic, np.zeros_like(best))
    codes, uniq = pd.factorize(d["patient_id"])
    d["pc"] = codes
    per = d.groupby("pc").agg(u_obs=("u_obs", "sum"), u_best=("u_best", "sum"), u_none=("u_none", "sum"))
    # patient-level detection flags
    pl_df = d.assign(t_s=t_s)
    win = pl_df[septic & (tt >= DT_EARLY) & (tt <= DT_LATE)]
    det = win.groupby("pc")["pred"].max()
    sep_codes = np.unique(codes[septic])
    per["septic"] = False
    per.loc[sep_codes, "septic"] = True
    per["detected"] = 0
    per.loc[det.index, "detected"] = det.to_numpy()
    per["alerted_ever"] = d.groupby("pc")["pred"].max()
    rows_of = d.groupby("pc").indices
    y, p = d["SepsisLabel"].to_numpy(), d["p"].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(per)
    out = {k: [] for k in ("auroc", "auprc", "utility", "patient_sensitivity", "patient_specificity")}
    for b in range(B):
        s = rng.integers(0, n, n)
        idx = np.concatenate([rows_of[i] for i in s])
        out["auroc"].append(roc_auc_score(y[idx], p[idx]))
        out["auprc"].append(average_precision_score(y[idx], p[idx]))
        ps = per.iloc[s]
        out["utility"].append((ps.u_obs.sum() - ps.u_none.sum()) / (ps.u_best.sum() - ps.u_none.sum()))
        sp = ps[ps.septic]; ns = ps[~ps.septic]
        out["patient_sensitivity"].append(sp.detected.mean())
        out["patient_specificity"].append(1 - ns.alerted_ever.mean())
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


# ---------------------------------------------------------------- main
def main():
    cfg = json.load(open(MODELS / "config.json"))
    cols, thr_main = cfg["features"], cfg["threshold"]
    feats = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    tr, va, te = split_patients(feats)
    m_tr, m_va, m_te = (feats["patient_id"].isin(s).to_numpy() for s in (tr, va, te))
    dva, dte = feats.loc[m_va].reset_index(drop=True), feats.loc[m_te].reset_index(drop=True)
    y_tr, y_va = feats.loc[m_tr, "SepsisLabel"].to_numpy(), dva["SepsisLabel"].to_numpy()
    R = {}

    # ---------- baselines
    base = []
    for name, col in (("SIRS criteria (rule: ≥2 of 4)", "sirs_score"), ("qSOFA-partial (rule: ≥1 of 2)", "qsofa_partial")):
        pv, pt = dva[col].fillna(0).to_numpy(), dte[col].fillna(0).to_numpy()
        rule_thr = 2 if "SIRS" in name else 1
        pred = (pt >= rule_thr).astype(int)
        pl = patient_level(dte, pred)
        r = {"auroc": roc_auc_score(dte["SepsisLabel"], pt), "auprc": average_precision_score(dte["SepsisLabel"], pt),
             "utility": utility_score(dte, pred), "patient_sensitivity": pl["patient_sensitivity"],
             "patient_specificity": pl["patient_specificity"], "pct_detected_ge6h_early": pl["pct_detected_ge6h_early"],
             "ece": None, "threshold": rule_thr}
        base.append({"model": name, **r})
        log(name, round(r["auroc"], 3), round(r["utility"], 3))
    # logistic regression on median-imputed, standardized features (+ missing indicators)
    t = time.time()
    med = feats.loc[m_tr, cols].median()

    def prep(X):
        Xi = X.fillna(med).fillna(0)
        return Xi.to_numpy(np.float32)
    sc = StandardScaler().fit(prep(feats.loc[m_tr, cols]))
    rng = np.random.default_rng(0)
    sub = rng.choice(np.where(m_tr)[0], 400_000, replace=False)
    lr = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(prep(feats.loc[sub, cols])), feats.loc[sub, "SepsisLabel"])
    lv = lr.predict_proba(sc.transform(prep(dva[cols])))[:, 1]
    lt = lr.predict_proba(sc.transform(prep(dte[cols])))[:, 1]
    iso = iso_fit(lv, y_va); thr, _ = best_threshold(dva, iso.predict(lv))
    base.append({"model": "Logistic regression (+ isotonic)", **summarize(dte, iso.predict(lt), thr)})
    log("LR", round(base[-1]["auroc"], 3), round(base[-1]["utility"], 3), f"{time.time()-t:.0f}s")
    # single LightGBM / ensemble raw / ensemble calibrated
    models = [lgb.Booster(model_file=str(MODELS / f"lgb_seed{s}.txt")) for s in cfg["seeds"]]
    Pv = np.column_stack([m.predict(dva[cols]) for m in models])
    Pt = np.column_stack([m.predict(dte[cols]) for m in models])
    thr_s, _ = best_threshold(dva, Pv[:, 0])
    base.append({"model": "Single LightGBM (uncalibrated)", **summarize(dte, Pt[:, 0], thr_s)})
    thr_e, _ = best_threshold(dva, Pv.mean(1))
    base.append({"model": "5-model LightGBM ensemble (uncalibrated)", **summarize(dte, Pt.mean(1), thr_e)})
    iso_e = pickle.load(open(MODELS / "calibrator.pkl", "rb"))
    cal_t = iso_e.predict(Pt.mean(1))
    base.append({"model": "Ensemble + isotonic calibration (SepsisShield model)", **summarize(dte, cal_t, thr_main)})
    R["baselines"] = base
    log("baselines done")

    # ---------- bootstrap CIs for the SepsisShield model and LR
    R["bootstrap_ci"] = {"sepsisshield": bootstrap_ci(dte, cal_t, thr_main),
                         "B": 500, "unit": "patient"}
    log("bootstrap", R["bootstrap_ci"]["sepsisshield"])

    # ---------- threshold sweep (test) + where validation chose
    grid = np.round(np.concatenate([np.arange(0.01, 0.1, 0.005), np.arange(0.1, 0.31, 0.02)]), 3)
    sw = []
    for g in grid:
        pred = (cal_t >= g).astype(int)
        pl = patient_level(dte, pred)
        sw.append({"threshold": float(g), "utility_test": utility_score(dte, pred),
                   "utility_val": utility_score(dva, (iso_e.predict(Pv.mean(1)) >= g).astype(int)),
                   "patient_sensitivity": pl["patient_sensitivity"], "patient_specificity": pl["patient_specificity"],
                   "false_alert_hours_per_100_patient_days": pl["false_alert_hours_per_100_patient_days"]})
    R["threshold_sweep"] = {"chosen": thr_main, "rows": sw}
    log("sweep done")

    # ---------- cross-hospital recalibration
    rc = []
    for src, dst in (("A", "B"), ("B", "A")):
        s = feats[feats["hospital"] == src]
        pat = s.groupby("patient_id", observed=True)["SepsisLabel"].max()
        trp, vap = train_test_split(np.asarray(pat.index, dtype=object), test_size=0.15,
                                    stratify=pat.to_numpy(), random_state=0)
        sv = s[s["patient_id"].isin(set(vap))].reset_index(drop=True)
        m = lgb.Booster(model_file=str(MODELS / f"lgb_train{src}.txt"))
        pv = m.predict(sv[cols]); iso_s = iso_fit(pv, sv["SepsisLabel"])
        thr_s, _ = best_threshold(sv, iso_s.predict(pv))
        d = feats[feats["hospital"] == dst]
        patd = d.groupby("patient_id", observed=True)["SepsisLabel"].max()
        loc, ev = train_test_split(np.asarray(patd.index, dtype=object), test_size=0.8,
                                   stratify=patd.to_numpy(), random_state=0)
        dl = d[d["patient_id"].isin(set(loc))].reset_index(drop=True)
        de = d[d["patient_id"].isin(set(ev))].reset_index(drop=True)
        pe_raw, pl_raw = m.predict(de[cols]), m.predict(dl[cols])
        as_is = summarize(de, iso_s.predict(pe_raw), thr_s)
        iso_l = iso_fit(pl_raw, dl["SepsisLabel"]); thr_l, _ = best_threshold(dl, iso_l.predict(pl_raw))
        local = summarize(de, iso_l.predict(pe_raw), thr_l)
        rc.append({"transfer": f"{src}→{dst}", "as_is": as_is, "recalibrated": local,
                   "n_local_patients": int(len(loc)), "n_eval_patients": int(len(ev))})
        log(f"recal {src}->{dst}: utility {as_is['utility']:.3f} -> {local['utility']:.3f}; "
            f"ECE {as_is['ece']*100:.2f} -> {local['ece']*100:.2f} pp; spec {as_is['patient_specificity']:.2f} -> {local['patient_specificity']:.2f}")
    R["recalibration"] = rc

    # ---------- subgroup table enrichment (n septic)
    tp = pd.read_parquet(RES / "test_predictions.parquet")
    R["subgroup_counts"] = {
        "hospital": tp.groupby("hospital", observed=True).apply(lambda g: int(g.groupby("patient_id").SepsisLabel.max().sum())).to_dict()}
    json.dump(R, open(RES / "experiments2.json", "w"), indent=1, default=float)
    log("saved")


if __name__ == "__main__":
    main()
