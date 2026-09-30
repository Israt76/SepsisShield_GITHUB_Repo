"""Trust-aware abstention: does the integrity layer stop corrupted predictions from reaching the clinician silently?

Policy (research prototype):
    HIGH     -> prediction shown normally
    REDUCED  -> prediction shown + "verify inputs" warning
    LOW      -> prediction WITHHELD ("requires data verification"); research score kept for audit

A *dangerous failure* is a patient-hour inside the corruption window where the fault flipped the alert decision
AND the new decision is wrong:
    suppressed alert   clean -> alert, corrupted -> no alert, in a septic patient's useful window (12 h before .. 3 h after onset)
    spurious alert     clean -> no alert, corrupted -> alert, OUTSIDE any useful window (non-septic patient, or septic
                       patient more than 12 h before / 3 h after onset)
New alerts that a fault happens to create inside a septic patient's useful window are not counted as dangerous
(they are reported separately as `spurious_in_useful_window`).

Without the integrity layer every dangerous failure reaches the clinician as a normal-looking prediction.
With it, a failure is
    withheld  (trust LOW)       -> the wrong prediction is never shown as actionable
    flagged   (trust REDUCED)   -> shown with a verification warning
    silent    (trust HIGH)      -> reaches the clinician unflagged

Dangerous Failure Coverage (DFC) = (withheld + flagged) / all dangerous failures, reported with numerator/denominator.
Cost on clean data = share of clean predictions withheld / flagged, and share of correct clean sepsis alerts withheld.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd

from metrics import onset_hour
from benchmark_integrity import corrupt
import pipeline

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
KINDS = ["temp_fahrenheit", "lab_unit_error", "sensor_artifact", "frozen_feed", "masking_attack", "measurement_noise"]
ACCIDENTAL = KINDS[:4]


def classify(clean, corr, m):
    ts = onset_hour(clean)
    t_s = clean["patient_id"].map(ts).to_numpy(float)
    septic = ~np.isnan(t_s)
    h = clean["hour"].to_numpy()
    in_win = septic & (h >= t_s - 12) & (h <= t_s + 3)
    ca, xa = clean["alert"].to_numpy(), corr["alert"].to_numpy()
    sup = m & in_win & ca & ~xa
    spu = m & ~in_win & ~ca & xa
    lucky = m & in_win & ~ca & xa
    tr, ctr = corr["trust"].to_numpy(), clean["trust"].to_numpy()
    out = {}
    for name, mask in (("suppressed", sup), ("spurious", spu), ("all", sup | spu), ("spurious_in_useful_window", lucky)):
        n = int(mask.sum())
        out[name] = {"n": n, "withheld": int((mask & (tr == "LOW")).sum()),
                     "flagged": int((mask & (tr == "REDUCED")).sum()), "silent": int((mask & (tr == "HIGH")).sum()),
                     # covered only because of flags the fault itself caused (clean trust at that hour was HIGH)
                     "covered_attributable": int((mask & (tr != "HIGH") & (ctr == "HIGH")).sum())}
    return out


def main(cohort="test", n_nonseptic=2500, seed=7):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    raw = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    if cohort == "test":
        test_ids = set(pd.read_parquet(RES / "test_patients.parquet")["patient_id"])
    else:  # validation cohort: never used for any integrity-layer design decision
        test_ids = set(pd.read_parquet(RES / "val_predictions.parquet", columns=["patient_id"])["patient_id"])
    raw = raw[raw["patient_id"].isin(test_ids)]
    sep = raw.groupby("patient_id")["SepsisLabel"].max()
    keep = set(sep[sep == 1].index) | set(rng.choice(sep[sep == 0].index.to_numpy(), n_nonseptic, replace=False))
    raw = raw[raw["patient_id"].isin(keep)].sort_values(["patient_id", "hour"]).reset_index(drop=True)
    clean, _ = pipeline.score(raw)

    # cost on clean data
    ts = onset_hour(clean)
    t_s = clean["patient_id"].map(ts).to_numpy(float)
    h = clean["hour"].to_numpy()
    in_win = ~np.isnan(t_s) & (h >= t_s - 12) & (h <= t_s + 3)
    true_alerts = in_win & clean["alert"].to_numpy()
    tr = clean["trust"].to_numpy()
    cost = {"hours": int(len(clean)),
            "withheld_pct": float((tr == "LOW").mean()), "flagged_pct": float((tr == "REDUCED").mean()),
            "true_alert_hours": int(true_alerts.sum()),
            "true_alerts_withheld": int((true_alerts & (tr == "LOW")).sum()),
            "true_alerts_flagged": int((true_alerts & (tr == "REDUCED")).sum())}
    # does every septic patient detected on clean data still get at least one SHOWN (non-withheld) alert?
    d = clean.assign(in_win=in_win, shown=clean["alert"] & (clean["trust"] != "LOW"))
    det = d[d.in_win].groupby("patient_id").agg(alert=("alert", "max"), shown=("shown", "max"))
    cost["septic_detected"] = int(det.alert.sum())
    cost["septic_detected_with_shown_alert"] = int((det.alert & det.shown).sum())

    # unbiased clean cost on the FULL held-out test set (benchmark sample oversamples septic patients)
    full = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    full = full[full["patient_id"].isin(test_ids)].sort_values(["patient_id", "hour"]).reset_index(drop=True)
    fs, _ = pipeline.score(full)
    ftr = fs["trust"]
    fsep = fs.groupby("patient_id")["SepsisLabel"].transform("max") == 1
    ever_low = (ftr == "LOW").groupby(fs["patient_id"]).any()
    ever_low_ns = ever_low[~fsep.groupby(fs["patient_id"]).first()]
    cost_full = {"patients": int(fs["patient_id"].nunique()), "hours": int(len(fs)),
                 "withheld_hours": int((ftr == "LOW").sum()), "withheld_pct": float((ftr == "LOW").mean()),
                 "flagged_pct": float((ftr == "REDUCED").mean()),
                 "patients_ever_withheld_pct": float(ever_low.mean()),
                 "nonseptic_patients_ever_withheld_pct": float(ever_low_ns.mean()),
                 "median_withheld_hours_per_affected_patient": float((ftr == "LOW").groupby(fs["patient_id"]).sum()[ever_low].median())}
    res = {"policy": "LOW=withhold, REDUCED=warn, HIGH=show", "n_patients": len(keep), "clean_cost": cost,
           "clean_cost_full_test": cost_full, "scenarios": []}
    for kind in KINDS:  # same RNG stream as benchmark_integrity.py -> identical corruption windows
        corr_raw, m = corrupt(raw, kind, rng)
        corr, _ = pipeline.score(corr_raw)
        c = classify(clean, corr, m)
        res["scenarios"].append({"corruption": kind, **c})
        a = c["all"]
        print(time.strftime("%H:%M:%S"), kind, a, flush=True)

    def agg(kinds, key):
        tot = {k: sum(s[key][k] for s in res["scenarios"] if s["corruption"] in kinds)
               for k in ("n", "withheld", "flagged", "silent", "covered_attributable")}
        tot["attributable_coverage"] = tot["covered_attributable"] / tot["n"] if tot["n"] else float("nan")
        tot["coverage"] = (tot["withheld"] + tot["flagged"]) / tot["n"] if tot["n"] else float("nan")
        tot["withheld_share"] = tot["withheld"] / tot["n"] if tot["n"] else float("nan")
        return tot
    res["definition"] = "flip AND wrong: suppressed timely alert, or new alert outside any useful window"
    res["dfc_accidental"] = {k: agg(ACCIDENTAL, k) for k in ("suppressed", "spurious", "all")}
    res["dfc_masking"] = {k: agg(["masking_attack"], k) for k in ("suppressed", "spurious", "all")}
    res["dfc_noise_control"] = {k: agg(["measurement_noise"], k) for k in ("suppressed", "spurious", "all")}
    res["runtime_s"] = time.time() - t0
    res["cohort"] = cohort
    json.dump(res, open(RES / ("abstention.json" if cohort == "test" else f"abstention_{cohort}.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("clean_cost", "dfc_accidental", "dfc_masking")}, indent=1))


if __name__ == "__main__":
    import sys
    c = sys.argv[1] if len(sys.argv) > 1 else "test"
    main(c, n_nonseptic=2500 if c == "test" else 1250)
