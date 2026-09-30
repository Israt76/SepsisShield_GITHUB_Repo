"""Corruption benchmark: does the integrity layer catch the input failures that break the model?

For held-out test patients we inject six realistic input failures into the RAW hourly data,
re-run the full pipeline (features -> ensemble -> calibration -> integrity checks) and measure:
  * how much each corruption damages the model (AUROC / utility / sepsis cases missed)
  * how often the integrity layer flags the corrupted hours (detection rate)
  * how often it flags clean hours (false-flag rate)
  * the key safety number: of the sepsis patients whose alert was SUPPRESSED by the corruption,
    how many were shown to the clinician with a LOW/REDUCED-trust warning instead of a silent all-clear

For septic patients the corruption window is placed in the clinically decisive period
(12h before to 3h after onset) — the worst case. For non-septic patients it is placed at random.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd

from data import VITALS, LABS
from metrics import onset_hour, hourly, utility_score, patient_level
import integrity
import pipeline

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
WIN = 10


def corrupt(raw: pd.DataFrame, kind: str, rng: np.random.Generator):
    raw = raw.copy()
    num = VITALS + LABS
    raw[num] = raw[num].astype("float64")
    ts = onset_hour(raw)
    mask = np.zeros(len(raw), dtype=bool)
    for pid, idx in raw.groupby("patient_id", sort=False).indices.items():
        n = len(idx)
        if n < 4:
            continue
        t_s = ts.get(pid, np.nan)
        if not np.isnan(t_s):
            lo, hi = max(0, int(t_s) - 12), max(1, min(n - 1, int(t_s) + 3) - WIN + 1)
            start = rng.integers(lo, max(lo + 1, hi))
        else:
            start = rng.integers(0, max(1, n - WIN))
        w = idx[start:start + WIN]
        mask[w] = True
    m = mask
    if kind == "temp_fahrenheit":          # thermometer / interface reporting °F into a °C field
        raw.loc[m, "Temp"] = raw.loc[m, "Temp"] * 1.8 + 32
    elif kind == "lab_unit_error":         # SI/conventional unit mix-up from an external lab
        raw.loc[m, "Creatinine"] *= 88.4   # mg/dL -> µmol/L
        raw.loc[m, "Glucose"] /= 18.0      # mg/dL -> mmol/L
        raw.loc[m, "Lactate"] *= 9.0       # mmol/L -> mg/dL
    elif kind == "sensor_artifact":        # motion/lead-off artefacts on monitor
        hit = m & (rng.random(len(raw)) < 0.5)
        raw.loc[hit, "HR"] = rng.uniform(190, 240, hit.sum()).round()
        raw.loc[hit, "SBP"] = rng.uniform(25, 45, hit.sum()).round()
    elif kind == "frozen_feed":            # interface stops updating; last values copied forward
        for c in ["HR", "SBP", "MAP", "DBP", "Resp", "O2Sat", "Temp"]:
            first = raw[c].where(m).groupby(raw["patient_id"]).transform("first")
            prior = raw.groupby("patient_id")[c].ffill()
            fill = first.fillna(prior)
            raw.loc[m, c] = fill[m]
    elif kind == "masking_attack":         # deliberate overwrite to make a deteriorating patient look stable
        raw.loc[m, "HR"] -= 25
        raw.loc[m, "Resp"] -= 8
        raw.loc[m, "Temp"] -= 1.2
        raw.loc[m, "SBP"] += 20
        raw.loc[m, "MAP"] += 15
        raw.loc[m, "Lactate"] = raw.loc[m, "Lactate"].clip(upper=1.5)
        raw.loc[m, "WBC"] = raw.loc[m, "WBC"].clip(5, 11)
    elif kind == "measurement_noise":      # benign control: ordinary noise, should NOT be flagged
        for c in ["HR", "SBP", "MAP", "DBP", "Resp"]:
            sd = raw[c].std() * 0.10
            raw.loc[m, c] += rng.normal(0, sd, m.sum())
    return raw, m


def detected_patients(scored: pd.DataFrame, thr_hours=None):
    ts = onset_hour(scored)
    d = scored.assign(t_s=scored["patient_id"].map(ts))
    s = d[d["t_s"].notna()]
    w = s[(s["hour"] >= s["t_s"] - 12) & (s["hour"] <= s["t_s"] + 3)]
    return set(w.loc[w["alert"], "patient_id"])


def summarize(clean, corr, m, kind):
    y = clean["SepsisLabel"].to_numpy()
    changed = m
    r = {"corruption": kind}
    hc, hx = hourly(y, clean["risk"].to_numpy()), hourly(y, corr["risk"].to_numpy())
    r["auroc_clean"], r["auroc_corrupt"] = hc["auroc"], hx["auroc"]
    r["utility_clean"] = utility_score(clean, clean["alert"].to_numpy().astype(int))
    r["utility_corrupt"] = utility_score(corr, corr["alert"].to_numpy().astype(int))
    # utility if the system withholds a verdict (no alert decision) whenever trust is LOW
    r["hour_detect_rate"] = float((corr.loc[changed, "trust"] != "HIGH").mean())
    r["hour_low_rate"] = float((corr.loc[changed, "trust"] == "LOW").mean())
    pw = corr.loc[changed].groupby("patient_id")["trust"].apply(lambda t: (t != "HIGH").any())
    r["patient_detect_rate"] = float(pw.mean())
    # matched baseline: same hours, same patients, but CLEAN data
    pc = clean.loc[changed].groupby("patient_id")["trust"].apply(lambda t: (t != "HIGH").any())
    r["clean_same_window_flag_rate"] = float(pc.mean())
    r["clean_same_window_low_rate"] = float(clean.loc[changed].groupby("patient_id")["trust"].apply(lambda t: (t == "LOW").any()).mean())
    # sepsis cases suppressed by the corruption
    det_c, det_x = detected_patients(clean), detected_patients(corr)
    suppressed = det_c - det_x
    r["septic_detected_clean"], r["septic_detected_corrupt"] = len(det_c), len(det_x)
    r["alerts_suppressed"] = len(suppressed)
    if suppressed:
        warned = corr[corr["patient_id"].isin(suppressed) & changed].groupby("patient_id")["trust"].apply(
            lambda t: (t != "HIGH").any())
        r["suppressed_but_warned"] = int(warned.sum())
        r["suppressed_warned_pct"] = float(warned.mean())
    else:
        r["suppressed_but_warned"], r["suppressed_warned_pct"] = 0, float("nan")
    # false alarms created by corruption on non-septic patients
    ns = clean.groupby("patient_id")["SepsisLabel"].max() == 0
    ns_ids = set(ns[ns].index)
    fa_c = set(clean.loc[clean["alert"] & clean["patient_id"].isin(ns_ids), "patient_id"])
    fa_x = set(corr.loc[corr["alert"] & corr["patient_id"].isin(ns_ids), "patient_id"])
    new_fa = fa_x - fa_c
    r["false_alarms_created"] = len(new_fa)
    if new_fa:
        warned = corr[corr["patient_id"].isin(new_fa) & changed].groupby("patient_id")["trust"].apply(
            lambda t: (t != "HIGH").any())
        r["created_fa_warned_pct"] = float(warned.mean())
    else:
        r["created_fa_warned_pct"] = float("nan")
    return r


def main(n_nonseptic=2500, seed=7):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    raw = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    test_ids = set(pd.read_parquet(RES / "test_patients.parquet")["patient_id"])
    raw = raw[raw["patient_id"].isin(test_ids)]
    sep = raw.groupby("patient_id")["SepsisLabel"].max()
    keep = set(sep[sep == 1].index) | set(rng.choice(sep[sep == 0].index.to_numpy(), n_nonseptic, replace=False))
    raw = raw[raw["patient_id"].isin(keep)].sort_values(["patient_id", "hour"]).reset_index(drop=True)
    clean, _ = pipeline.score(raw)
    res = {"n_patients": int(len(keep)), "n_septic": int((sep.loc[list(keep)] == 1).sum())}
    res["clean_false_flag_rate_hours"] = float((clean["trust"] != "HIGH").mean())
    res["clean_low_rate_hours"] = float((clean["trust"] == "LOW").mean())
    res["clean_patients_ever_low"] = float(clean.groupby("patient_id")["trust"].apply(lambda t: (t == "LOW").any()).mean())
    res["clean_patients_ever_flagged"] = float(clean.groupby("patient_id")["trust"].apply(lambda t: (t != "HIGH").any()).mean())
    rows = []
    for kind in ["temp_fahrenheit", "lab_unit_error", "sensor_artifact", "frozen_feed",
                 "masking_attack", "measurement_noise"]:
        corr_raw, m = corrupt(raw, kind, rng)
        corr, _ = pipeline.score(corr_raw)
        rows.append(summarize(clean, corr, m, kind))
        print(time.strftime("%H:%M:%S"), json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                      for k, v in rows[-1].items()}), flush=True)
    res["corruptions"] = rows
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(RES / "integrity_benchmark.json", "w"), indent=1)
    return res


def real_artifacts():
    """How often does the ORIGINAL (uncorrupted) PhysioNet data already contain impossible values?"""
    raw = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet").sort_values(["patient_id", "hour"]).reset_index(drop=True)
    chk = integrity.check(raw)
    out = {"hours": int(len(raw)),
           "hours_implausible": int(chk["flag_implausible"].sum()),
           "patients_with_implausible": int(raw.loc[chk["flag_implausible"], "patient_id"].nunique()),
           "hours_inconsistent": int(chk["flag_inconsistent"].sum()),
           "hours_flatline": int(chk["flag_flatline"].sum())}
    vc = chk.loc[chk["flag_implausible"], "implausible_vars"].str.split().explode().value_counts()
    out["top_implausible_vars"] = vc.head(8).to_dict()
    return out


if __name__ == "__main__":
    r = main()
    r["real_data_artifacts"] = real_artifacts()
    json.dump(r, open(RES / "integrity_benchmark.json", "w"), indent=1)
    print(json.dumps(r["real_data_artifacts"], indent=1))
