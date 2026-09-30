"""Train SepsisShield models and run all validation experiments.

Experiments
  main        pooled A+B, patient-level 70/10/20 split, 5-seed LightGBM ensemble,
              isotonic calibration, utility-optimal threshold chosen on validation only
  cross_A2B   train on Hospital A, test on unseen Hospital B (external validation)
  cross_B2A   train on Hospital B, test on unseen Hospital A
  ablation    incremental feature groups (single model, same split as main)
  subgroups   main test-set metrics by age, sex, hospital, ICU type
"""
from pathlib import Path
import json, pickle, time, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from features import feature_columns, groups_for
from metrics import full_report, best_threshold, hourly, patient_level, utility_score

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "processed" / "features.parquet"
MODELS = ROOT / "models"
RES = ROOT / "results"
SEEDS = [11, 22, 33, 44, 55]

# selected on the validation set only (see results/logs/log_tune.txt); the test set was never used for tuning
PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=31, min_data_in_leaf=500,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              metric="auc", num_threads=2, verbose=-1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def split_patients(df: pd.DataFrame, seed=0, test=0.2, val=0.1):
    pat = df.groupby("patient_id", observed=True).agg(h=("hospital", "first"), s=("SepsisLabel", "max"))
    ids = np.asarray(pat.index, dtype=object)
    strat = (pat["h"].astype(str) + pat["s"].astype(str)).to_numpy()
    tr, te, s_tr, _ = train_test_split(ids, strat, test_size=test, stratify=strat, random_state=seed)
    tr, va = train_test_split(tr, test_size=val / (1 - test), stratify=s_tr, random_state=seed)
    return set(tr), set(va), set(te)


def fit(X_tr, y_tr, X_va, y_va, seed, rounds=3000, cols=None):
    p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    dtr = lgb.Dataset(X_tr, y_tr, free_raw_data=True)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    return lgb.train(p, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(150, verbose=False)])


def predict_ens(models, X):
    P = np.column_stack([m.predict(X, num_iteration=m.best_iteration) for m in models])
    return P.mean(1), P.std(1)


def run_main(df, cols):
    tr, va, te = split_patients(df)
    m_tr, m_va, m_te = (df["patient_id"].isin(s).to_numpy() for s in (tr, va, te))
    X, y = df[cols], df["SepsisLabel"].to_numpy()
    models = []
    for s in SEEDS:
        t = time.time()
        m = fit(X[m_tr], y[m_tr], X[m_va], y[m_va], s)
        models.append(m)
        log(f"main seed={s} iters={m.best_iteration} {time.time()-t:.0f}s")
        m.save_model(str(MODELS / f"lgb_seed{s}.txt"), num_iteration=m.best_iteration)

    raw_va, sd_va = predict_ens(models, X[m_va])
    raw_te, sd_te = predict_ens(models, X[m_te])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_va, y[m_va])
    cal_va, cal_te = iso.predict(raw_va), iso.predict(raw_te)
    dva, dte = df.loc[m_va], df.loc[m_te]
    thr, u_va = best_threshold(dva, cal_va)
    log(f"threshold={thr:.3f} (val utility {u_va:.3f})")

    rep = {
        "calibrated": full_report(dte, cal_te, thr),
        "uncalibrated_hourly": hourly(y[m_te], raw_te),
        "val_utility": u_va,
        "single_model_test": {},
        "n_patients": {"train": len(tr), "val": len(va), "test": len(te)},
    }
    # single model vs ensemble (does the ensemble add anything?)
    single = models[0].predict(X[m_te], num_iteration=models[0].best_iteration)
    rep["single_model_test"] = hourly(y[m_te], single)

    with open(MODELS / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    json.dump({"threshold": thr, "features": cols, "seeds": SEEDS},
              open(MODELS / "config.json", "w"), indent=1)
    out = dte[["patient_id", "hospital", "hour", "SepsisLabel", "Age", "Gender", "Unit1", "Unit2"]].copy()
    out["prob_raw"], out["prob"], out["ens_std"] = raw_te, cal_te, sd_te
    out.to_parquet(RES / "test_predictions.parquet", index=False)
    vout = dva[["patient_id", "hour", "SepsisLabel"]].copy()
    vout["prob_raw"], vout["prob"], vout["ens_std"] = raw_va, cal_va, sd_va
    vout.to_parquet(RES / "val_predictions.parquet", index=False)
    pd.Series(sorted(te)).to_frame("patient_id").to_parquet(RES / "test_patients.parquet")
    return rep, models, (tr, va, te)


def run_cross(df, cols, src, dst):
    s = df[df["hospital"] == src]
    pat = s.groupby("patient_id", observed=True)["SepsisLabel"].max()
    tr, _va = train_test_split(np.asarray(pat.index, dtype=object), test_size=0.15,
                              stratify=pat.to_numpy(), random_state=0)
    tr = set(tr)
    m_tr = s["patient_id"].isin(tr).to_numpy(); m_va = ~m_tr
    m = fit(s.loc[m_tr, cols], s.loc[m_tr, "SepsisLabel"], s.loc[m_va, cols], s.loc[m_va, "SepsisLabel"], 11)
    raw_va = m.predict(s.loc[m_va, cols], num_iteration=m.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_va, s.loc[m_va, "SepsisLabel"])
    thr, _u = best_threshold(s.loc[m_va], iso.predict(raw_va))
    d = df[df["hospital"] == dst]
    p = iso.predict(m.predict(d[cols], num_iteration=m.best_iteration))
    rep = full_report(d, p, thr)
    # internal (same-hospital) validation score for comparison
    rep["internal_val"] = full_report(s.loc[m_va], iso.predict(raw_va), thr)
    m.save_model(str(MODELS / f"lgb_train{src}.txt"), num_iteration=m.best_iteration)
    log(f"cross {src}->{dst}: AUROC {rep['auroc']:.3f} utility {rep['utility']:.3f} "
        f"(internal {rep['internal_val']['auroc']:.3f}/{rep['internal_val']['utility']:.3f})")
    return rep


def run_ablation(df, cols, splits):
    tr, va, te = splits
    m_tr, m_va, m_te = (df["patient_id"].isin(s).to_numpy() for s in (tr, va, te))
    g = groups_for(cols)
    order = [("Demographics + vitals (raw)", ["demographics", "vitals"]),
             ("+ labs", ["labs"]), ("+ missingness patterns", ["missingness"]),
             ("+ temporal trends", ["trends"]), ("+ clinical composites (full)", ["composites"])]
    used, rows = [], []
    y = df["SepsisLabel"].to_numpy()
    for name, add in order:
        for k in add:
            used += g[k]
        m = fit(df.loc[m_tr, used], y[m_tr], df.loc[m_va, used], y[m_va], 11)
        pv = m.predict(df.loc[m_va, used], num_iteration=m.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, y[m_va])
        thr, _ = best_threshold(df.loc[m_va], iso.predict(pv))
        pt = iso.predict(m.predict(df.loc[m_te, used], num_iteration=m.best_iteration))
        r = full_report(df.loc[m_te], pt, thr)
        rows.append({"config": name, "n_features": len(used), **{k: r[k] for k in
                     ("auroc", "auprc", "utility", "patient_sensitivity", "patient_specificity", "median_lead_time_h")}})
        log(f"ablation {name}: {rows[-1]['auroc']:.3f} {rows[-1]['utility']:.3f}")
    return rows


def run_subgroups(thr):
    d = pd.read_parquet(RES / "test_predictions.parquet")
    d["age_group"] = pd.cut(d["Age"], [0, 45, 65, 80, 200], labels=["<45", "45-64", "65-79", "80+"], right=False)
    d["sex"] = d["Gender"].map({0: "Female", 1: "Male"})
    d["icu"] = np.select([d["Unit1"] == 1, d["Unit2"] == 1], ["MICU", "SICU"], "Unknown")
    rows = []
    for col in ("hospital", "age_group", "sex", "icu"):
        for val, g in d.groupby(col, observed=True):
            y = g["SepsisLabel"].to_numpy()
            if y.sum() < 20:
                continue
            pred = (g["prob"] >= thr).astype(int).to_numpy()
            h = hourly(y, g["prob"].to_numpy()); pl = patient_level(g, pred)
            rows.append({"attribute": col, "group": str(val), "n_patients": g["patient_id"].nunique(),
                         "auroc": h["auroc"], "auprc": h["auprc"], "prevalence": h["prevalence"],
                         "mean_pred": float(g["prob"].mean()), "ece": h["ece"],
                         "utility": utility_score(g, pred),
                         "patient_sensitivity": pl["patient_sensitivity"],
                         "patient_specificity": pl["patient_specificity"]})
    return rows


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["main", "cross", "ablation", "subgroups"]
    df = pd.read_parquet(FEAT)
    cols = feature_columns(df)
    log(f"loaded {len(df):,} rows, {len(cols)} features")
    results = json.load(open(RES / "results.json")) if (RES / "results.json").exists() else {}
    splits = None
    if "main" in which:
        results["main"], _, splits = run_main(df, cols)
        json.dump(results, open(RES / "results.json", "w"), indent=1)
        log("main:", json.dumps(results["main"]["calibrated"], indent=0))
    if "cross" in which:
        results["cross_A2B"] = run_cross(df, cols, "A", "B")
        results["cross_B2A"] = run_cross(df, cols, "B", "A")
        json.dump(results, open(RES / "results.json", "w"), indent=1)
    if "ablation" in which:
        splits = splits or split_patients(df)
        results["ablation"] = run_ablation(df, cols, splits)
        json.dump(results, open(RES / "results.json", "w"), indent=1)
    if "subgroups" in which:
        thr = json.load(open(MODELS / "config.json"))["threshold"]
        results["subgroups"] = run_subgroups(thr)
        json.dump(results, open(RES / "results.json", "w"), indent=1)
    log("done")
