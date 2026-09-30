"""Causal (no look-ahead) hourly feature engineering.

Every feature at hour t uses only observations at hours <= t, so the model can
run in real time at the bedside.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from data import VITALS, LABS, LABEL

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "hourly.parquet"
OUT = ROOT / "data" / "processed" / "features.parquet"

RAW_VARS = VITALS + LABS
ROLL_VARS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "DBP"]
KEY_LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Bilirubin_total", "BUN"]
FEATURE_GROUPS: dict[str, list[str]] = {}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must contain one or more patients, sorted by (patient_id, hour)."""
    g = df.groupby("patient_id", sort=False)
    out = df[["patient_id", "hospital", "hour", "Age", "Gender", "Unit1",
              "Unit2", "HospAdmTime", "ICULOS"]].copy()
    if LABEL in df:
        out[LABEL] = df[LABEL]

    # 1. last observed value (forward fill within patient)
    ff = g[RAW_VARS].ffill()
    ff.columns = [f"{c}" for c in RAW_VARS]
    out[RAW_VARS] = ff.astype("float32")

    # 2. hours since last measurement (informative missingness)
    hour = df["hour"].to_numpy()
    hs_cols = {}
    for c in RAW_VARS:
        last = pd.Series(np.where(df[c].notna(), hour, np.nan), index=df.index)
        last = last.groupby(df["patient_id"], sort=False).ffill()
        hs_cols[f"{c}_hrs_since"] = (hour - last).astype("float32")
    # number of lab measurements so far (care intensity)
    meas = df[LABS].notna().sum(axis=1)
    hs_cols["n_labs_cum"] = meas.groupby(df["patient_id"], sort=False).cumsum().astype("float32")
    out = pd.concat([out, pd.DataFrame(hs_cols, index=df.index)], axis=1)

    # 3. rolling-window trends for vitals (6h and 12h)
    ffv = out[ROLL_VARS].copy()
    ffv["patient_id"] = df["patient_id"].to_numpy()
    gv = ffv.groupby("patient_id", sort=False)
    roll = {}
    for w in (6, 12):
        r = gv[ROLL_VARS].rolling(w, min_periods=1)
        mean, mn, mx, sd = r.mean(), r.min(), r.max(), r.std()
        for stat, frame in (("mean", mean), ("min", mn), ("max", mx), ("std", sd)):
            frame = frame.reset_index(level=0, drop=True).sort_index()
            for c in ROLL_VARS:
                roll[f"{c}_{stat}{w}"] = frame[c].to_numpy(dtype="float32")
        shifted = gv[ROLL_VARS].shift(w - 1)
        for c in ROLL_VARS:
            roll[f"{c}_diff{w}"] = (ffv[c] - shifted[c]).to_numpy(dtype="float32")
    out = pd.concat([out, pd.DataFrame(roll, index=df.index)], axis=1)

    # 4. lab trajectory: change since previous distinct measurement, running max
    lab = {}
    for c in KEY_LABS:
        obs = df[c]
        o_only = obs.dropna()
        prev_at_obs = o_only.groupby(df["patient_id"].loc[o_only.index], sort=False).shift(1)
        prev = prev_at_obs.reindex(df.index).groupby(df["patient_id"], sort=False).ffill()
        lab[f"{c}_delta"] = (out[c] - prev).astype("float32")
        lab[f"{c}_cummax"] = obs.groupby(df["patient_id"], sort=False).cummax().astype("float32")
    out = pd.concat([out, pd.DataFrame(lab, index=df.index)], axis=1)

    # 5. clinically motivated composites (SIRS / qSOFA / SOFA-style)
    o = out
    comp = pd.DataFrame(index=df.index)
    comp["shock_index"] = o["HR"] / o["SBP"]
    comp["pulse_pressure"] = o["SBP"] - o["DBP"]
    comp["bun_creat"] = o["BUN"] / o["Creatinine"]
    comp["sirs_hr"] = (o["HR"] > 90).astype("float32")
    comp["sirs_temp"] = ((o["Temp"] > 38) | (o["Temp"] < 36)).astype("float32")
    comp["sirs_resp"] = ((o["Resp"] > 20) | (o["PaCO2"] < 32)).astype("float32")
    comp["sirs_wbc"] = ((o["WBC"] > 12) | (o["WBC"] < 4)).astype("float32")
    comp["sirs_score"] = comp[["sirs_hr", "sirs_temp", "sirs_resp", "sirs_wbc"]].sum(axis=1)
    comp["qsofa_partial"] = ((o["Resp"] >= 22).astype(int) + (o["SBP"] <= 100).astype(int)).astype("float32")
    comp["sofa_cardio"] = (o["MAP"] < 70).astype("float32")
    comp["sofa_coag"] = pd.cut(o["Platelets"], [-np.inf, 20, 50, 100, 150, np.inf],
                               labels=[4, 3, 2, 1, 0]).astype("float32")
    comp["sofa_liver"] = pd.cut(o["Bilirubin_total"], [-np.inf, 1.2, 2, 6, 12, np.inf],
                                labels=[0, 1, 2, 3, 4]).astype("float32")
    comp["sofa_renal"] = pd.cut(o["Creatinine"], [-np.inf, 1.2, 2, 3.5, 5, np.inf],
                                labels=[0, 1, 2, 3, 4]).astype("float32")
    comp["sofa_resp_proxy"] = (o["O2Sat"] < 92).astype("float32")
    comp["sofa_partial"] = comp[["sofa_cardio", "sofa_coag", "sofa_liver",
                                 "sofa_renal", "sofa_resp_proxy"]].sum(axis=1, min_count=1)
    out = pd.concat([out, comp.astype("float32")], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)

    FEATURE_GROUPS.clear()
    FEATURE_GROUPS["demographics"] = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
    FEATURE_GROUPS["vitals"] = VITALS
    FEATURE_GROUPS["labs"] = LABS
    FEATURE_GROUPS["missingness"] = list(hs_cols)
    FEATURE_GROUPS["trends"] = list(roll) + list(lab)
    FEATURE_GROUPS["composites"] = list(comp.columns)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"patient_id", "hospital", "hour", LABEL}
    return [c for c in df.columns if c not in drop]


def groups_for(cols: list[str]) -> dict[str, list[str]]:
    """Recover feature groups from column names (usable without re-running make_features)."""
    grp = {"demographics": [], "vitals": [], "labs": [], "missingness": [],
           "trends": [], "composites": []}
    for c in cols:
        if c in ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"):
            grp["demographics"].append(c)
        elif c in VITALS:
            grp["vitals"].append(c)
        elif c in LABS:
            grp["labs"].append(c)
        elif c.endswith("_hrs_since") or c == "n_labs_cum":
            grp["missingness"].append(c)
        elif any(c.endswith(s) for s in ("_delta", "_cummax")) or any(
                f"_{s}" in c for s in ("mean6", "min6", "max6", "std6", "diff6",
                                       "mean12", "min12", "max12", "std12", "diff12")):
            grp["trends"].append(c)
        else:
            grp["composites"].append(c)
    return grp


if __name__ == "__main__":
    import time
    t = time.time()
    df = pd.read_parquet(SRC).sort_values(["patient_id", "hour"]).reset_index(drop=True)
    feats = make_features(df)
    feats.to_parquet(OUT, index=False)
    cols = feature_columns(feats)
    print(f"features={len(cols)} rows={len(feats):,} in {time.time()-t:.0f}s")
    for k, v in groups_for(cols).items():
        print(f"  {k}: {len(v)}")
