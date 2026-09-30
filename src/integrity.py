"""SepsisShield Input-Integrity Layer.

Decides, for every patient-hour, whether the INPUTS feeding the model can be trusted.
It runs on raw hourly measurements before feature engineering and is independent of
the prediction model, so a corrupted input cannot silently turn into a confident alert
(or a confident all-clear).

Checks
  1. plausibility   value outside physiologically possible limits (unit errors, sensor faults)
  2. consistency    internal contradictions (DBP >= SBP, MAP outside [DBP, SBP],
                    SaO2 vs O2Sat disagreement)
  3. jump           hour-to-hour change larger than the 99.95th percentile seen in clean
                    training data for that variable
  4. coordinated    several vitals move toward "normal" in the same hour (HR, Resp, Temp
     shift          down; SBP, MAP up) — summed clipped z-score vs the patient's own
                    baseline; a signature of manual overwrite / manipulation
  5. flatline       >= 3 core vitals identical to the decimal for >= 8 consecutive hours
                    (frozen feed / copied-forward values)
  6. model OOD      ensemble disagreement above the 99th percentile of validation
                    (applied later, in trust_level)

Output: per-hour flag columns + trust in {HIGH, REDUCED, LOW}.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from data import VITALS, LABS

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "models" / "integrity.json"

# Physiologically possible limits (deliberately wide: these are "cannot be real", not "abnormal")
PLAUSIBLE = {
    "HR": (20, 250), "O2Sat": (50, 100), "Temp": (30, 43), "SBP": (40, 280), "MAP": (20, 220),
    "DBP": (10, 200), "Resp": (3, 70), "EtCO2": (5, 100),
    "BaseExcess": (-30, 30), "HCO3": (3, 60), "FiO2": (0.21, 1.0), "pH": (6.5, 8.0),
    "PaCO2": (8, 150), "SaO2": (30, 100), "AST": (2, 20000), "BUN": (1, 300),
    "Alkalinephos": (5, 5000), "Calcium": (2, 20), "Chloride": (50, 150), "Creatinine": (0.1, 25),
    "Bilirubin_direct": (0, 50), "Glucose": (15, 1500), "Lactate": (0.1, 35), "Magnesium": (0.3, 10),
    "Phosphate": (0.3, 20), "Potassium": (1, 12), "Bilirubin_total": (0.05, 60),
    "TroponinI": (0, 500), "Hct": (5, 75), "Hgb": (1.5, 25), "PTT": (10, 250),
    "WBC": (0.05, 300), "Fibrinogen": (20, 2000), "Platelets": (1, 2000),
}
JUMP_VARS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
CORE = ["HR", "SBP", "Resp", "O2Sat", "MAP"]
# direction of a "reassuring" move for coordinated-shift detection (+1 up is reassuring, -1 down)
REASSURE = {"HR": -1, "Resp": -1, "Temp": -1, "SBP": +1, "MAP": +1}


def _prev_mean(frame: pd.DataFrame, pid: pd.Series, w: int = 6, minp: int = 3) -> pd.DataFrame:
    """Per-patient mean of the previous w hours (excluding the current hour)."""
    sh = frame.groupby(pid, sort=False).shift(1)
    r = sh.groupby(pid.to_numpy(), sort=False).rolling(w, min_periods=minp).mean()
    return r.reset_index(level=0, drop=True).reindex(frame.index)


def _moves(df: pd.DataFrame, pid: pd.Series) -> pd.DataFrame:
    """One-sided 'reassuring' move of each vital vs the patient's own previous-6h mean
    (positive = toward normal: HR/Resp/Temp down, SBP/MAP up)."""
    ffr = df.groupby(pid, sort=False)[list(REASSURE)].ffill()
    basef = _prev_mean(ffr, pid)
    return pd.DataFrame({c: sign * (df[c] - basef[c]) for c, sign in REASSURE.items()}, index=df.index)


def shift_score(moves: pd.DataFrame, scale: dict) -> np.ndarray:
    """Evidence of a coordinated normalising shift: sum over vitals of clipped z = move/scale in [0, 4].
    Clipping stops a single extreme value (handled by the jump check) from triggering it alone."""
    z = sum(np.clip((moves[c] / scale[c]).fillna(0).to_numpy(), 0, 4) for c in REASSURE)
    return z


def fit_thresholds(df: pd.DataFrame, q: float = 0.9995) -> dict:
    """Learn jump thresholds and coordinated-shift thresholds from clean training patients."""
    df = df.sort_values(["patient_id", "hour"])
    g = df.groupby("patient_id", sort=False)
    ff = g[JUMP_VARS].ffill()
    prev = ff.groupby(df["patient_id"], sort=False).shift(1)
    jumps = {}
    for c in JUMP_VARS:
        d = (df[c] - prev[c]).abs().dropna()
        jumps[c] = float(d.quantile(q))
    # Coordinated "reassuring" shift. Scale = robust spread of each vital's hour-to-baseline move in
    # clean training data; threshold = 99.75th percentile of the clean score (0.25% false-flag budget).
    # No attack data is used to set anything here.
    moves = _moves(df, df["patient_id"])
    scale = {c: float((moves[c].quantile(0.84) - moves[c].quantile(0.16)) / 2) for c in REASSURE}
    sc = shift_score(moves, scale)
    shift_thr = float(np.quantile(sc, 0.9975))
    rate = float((sc > shift_thr).mean())
    cfg = {"jump": jumps, "shift_scale": scale, "shift_thr": shift_thr, "shift_clean_flag_rate": rate,
           "plausible": PLAUSIBLE, "quantile": q}
    CFG.parent.mkdir(exist_ok=True)
    json.dump(cfg, open(CFG, "w"), indent=1)
    return cfg


def load_cfg() -> dict:
    return json.load(open(CFG))


def check(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Run all input checks. df: raw hourly rows sorted by (patient_id, hour)."""
    cfg = cfg or load_cfg()
    pid = df["patient_id"]
    out = pd.DataFrame(index=df.index)

    # 1. plausibility
    bad = np.zeros(len(df), dtype=bool)
    bad_vars = pd.Series([""] * len(df), index=df.index, dtype=object)
    # Calcium 0.8-2.0 is the signature of ionized Ca (mmol/L) entered in the total-Ca (mg/dL) field:
    # a known data-entry pattern (≈11% of calcium values in this dataset). Soft note, not "impossible".
    ca_note = df["Calcium"].between(0.8, 2.0)
    out["flag_unit_note"] = ca_note.to_numpy()
    for c, (lo, hi) in cfg["plausible"].items():
        m = (df[c] < lo) | (df[c] > hi)
        if c == "Calcium":
            m &= ~ca_note
        if m.any():
            bad |= m.to_numpy()
            bad_vars[m] = bad_vars[m] + c + " "
    out["flag_implausible"] = bad
    out["implausible_vars"] = bad_vars.str.strip()

    # 2. consistency
    # hard: physically impossible combinations within one blood-pressure reading
    cons = (df["DBP"] >= df["SBP"]) | (df["MAP"] > df["SBP"] + 5)
    out["flag_inconsistent"] = cons.fillna(False).to_numpy()
    # soft: discordance that also occurs legitimately (cuff vs arterial line, SpO2 vs SaO2 timing)
    soft = (df["MAP"] < df["DBP"] - 5) | ((df["SaO2"] - df["O2Sat"]).abs() > 25)
    out["flag_discordant"] = soft.fillna(False).to_numpy()

    # 3. jumps (vs last observed value)
    g = df.groupby(pid, sort=False)
    ff = g[JUMP_VARS].ffill()
    prev = ff.groupby(pid, sort=False).shift(1)
    jump = np.zeros(len(df), dtype=bool)
    jump_vars = pd.Series([""] * len(df), index=df.index, dtype=object)
    for c in JUMP_VARS:
        m = ((df[c] - prev[c]).abs() > cfg["jump"][c]).fillna(False)
        jump |= m.to_numpy()
        jump_vars[m] = jump_vars[m] + c + " "
    out["flag_jump"] = jump
    out["jump_vars"] = jump_vars.str.strip()

    # 4. coordinated reassuring shift across vitals
    sc = shift_score(_moves(df, pid), cfg["shift_scale"])
    out["shift_score"] = sc
    out["flag_coordinated_shift"] = sc > cfg["shift_thr"]

    # 5. flatline: core vitals unchanged for >= 8 consecutive observed hours
    same = np.ones(len(df), dtype=bool)
    for c in ["HR", "SBP", "Resp"]:
        v = df[c]
        same &= (v.notna() & (v == g[c].shift(1))).to_numpy()
    run = pd.Series(same.astype(int), index=df.index)
    blk = (run == 0).groupby(pid, sort=False).cumsum()
    streak = run.groupby([pid, blk], sort=False).cumsum()
    out["flag_flatline"] = (streak >= 7).to_numpy()  # 7 repeats = 8 identical hours

    # sticky window: a corrupted value keeps influencing forward-filled/rolling features,
    # so keep the patient flagged for 6h after any hard flag
    hard = out[["flag_implausible", "flag_inconsistent", "flag_jump",
                "flag_coordinated_shift", "flag_flatline"]].any(axis=1).astype(int)  # soft flags are not sticky
    last_hard = pd.Series(np.where(hard, df["hour"], np.nan), index=df.index).groupby(pid, sort=False).ffill()
    out["hours_since_flag"] = (df["hour"] - last_hard).to_numpy()
    out["input_alert"] = (out["hours_since_flag"] <= 6).fillna(False).to_numpy()
    return out


def trust_level(chk: pd.DataFrame, ens_std: np.ndarray, std_thr: float) -> np.ndarray:
    """Combine input checks with model disagreement into HIGH / REDUCED / LOW.
    ens_std is the ensemble spread on the LOG-ODDS scale, so it is not confounded with risk level."""
    # LOW only for evidence that the inputs cannot be right (or look deliberately normalised).
    # Abrupt jumps can be genuine deterioration, so they reduce trust rather than veto the score.
    low = chk["flag_implausible"] | chk["flag_inconsistent"] | chk["flag_coordinated_shift"]
    reduced = chk["input_alert"] | chk["flag_flatline"] | chk["flag_discordant"] | chk["flag_unit_note"] | \
        (ens_std > std_thr)
    return np.where(low, "LOW", np.where(reduced, "REDUCED", "HIGH"))


def explain_flags(row: pd.Series) -> list[str]:
    msgs = []
    if row.get("flag_implausible"):
        msgs.append(f"Physiologically impossible value(s): {row['implausible_vars']} — check units / sensor")
    if row.get("flag_inconsistent"):
        msgs.append("Internally inconsistent readings (e.g. diastolic ≥ systolic, MAP outside BP range)")
    if row.get("flag_jump"):
        msgs.append(f"Abrupt change beyond anything seen in training data: {row['jump_vars']}")
    if row.get("flag_coordinated_shift"):
        msgs.append("Several vitals shifted toward 'normal' in the same hour — possible overwrite or manipulation")
    if row.get("flag_discordant"):
        msgs.append("Readings from different devices disagree (cuff vs arterial line, or SpO₂ vs SaO₂)")
    if row.get("flag_unit_note"):
        msgs.append("Calcium looks like ionized Ca (mmol/L) entered as total Ca (mg/dL) — confirm units")
    if row.get("flag_flatline"):
        msgs.append("Core vitals identical for ≥8 hours — possible frozen or copied-forward feed")
    if not msgs and row.get("input_alert"):
        msgs.append(f"Input problem {int(row['hours_since_flag'])}h ago may still affect 6-12h trend features")
    return msgs


if __name__ == "__main__":
    # Fit thresholds on TRAINING patients only (never test), plus ensemble-disagreement
    # threshold from the validation predictions.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train import split_patients
    raw = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    tr, va, te = split_patients(raw)
    cfg = fit_thresholds(raw[raw["patient_id"].isin(tr)])
    # model-disagreement threshold: 99th percentile of log-odds ensemble spread on VALIDATION patients
    import lightgbm as lgb
    mcfg = json.load(open(ROOT / "models" / "config.json"))
    feats = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet", columns=["patient_id"] + mcfg["features"])
    Xv = feats[feats["patient_id"].isin(va)][mcfg["features"]]
    P = np.column_stack([lgb.Booster(model_file=str(ROOT / "models" / f"lgb_seed{sd}.txt")).predict(Xv)
                         for sd in mcfg["seeds"]])
    L = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
    cfg["ens_std_thr"] = float(np.quantile(L.std(1), 0.99))
    cfg["ens_std_scale"] = "logit"
    json.dump(cfg, open(CFG, "w"), indent=1)
    print(json.dumps({k: cfg[k] for k in ("jump", "shift_scale", "shift_thr", "shift_clean_flag_rate", "ens_std_thr")}, indent=1))
