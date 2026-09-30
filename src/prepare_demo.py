"""Pick representative held-out TEST patients for the dashboard (never training patients)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from metrics import onset_hour
import integrity

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main(seed=3):
    rng = np.random.default_rng(seed)
    thr = json.load(open(ROOT / "models" / "config.json"))["threshold"]
    te = pd.read_parquet(RES / "test_predictions.parquet")
    ts = onset_hour(te)
    los = te.groupby("patient_id")["hour"].max() + 1
    hosp = te.groupby("patient_id")["hospital"].first().astype(str)
    te["t_s"] = te["patient_id"].map(ts)
    w = te[te["t_s"].notna() & (te["prob"] >= thr) & (te["hour"] >= te["t_s"] - 12) & (te["hour"] <= te["t_s"] + 3)]
    lead = w.groupby("patient_id").apply(lambda g: (g["t_s"] - g["hour"]).max())
    septic = ts.dropna().index
    ever_alert = te.groupby("patient_id")["prob"].max() >= thr
    good_los = los[(los >= 30) & (los <= 140)].index

    def pick(ids, k):
        ids = [i for i in ids if i in set(good_los)]
        return list(rng.choice(ids, min(k, len(ids)), replace=False)) if ids else []

    early = pick(lead[lead >= 6].index[(ts.loc[lead[lead >= 6].index] >= 20).to_numpy()], 14)
    missed = pick([p for p in septic if p not in lead.index or lead[p] < 0], 8)
    stable = pick([p for p in ever_alert.index if not ever_alert[p] and p not in septic], 10)
    fa = pick([p for p in ever_alert.index if ever_alert[p] and p not in septic], 8)

    raw = pd.read_parquet(ROOT / "data" / "processed" / "hourly.parquet")
    raw = raw[raw["patient_id"].isin(set(te["patient_id"]))].sort_values(["patient_id", "hour"]).reset_index(drop=True)
    chk = integrity.check(raw)
    art_ids = raw.loc[chk["flag_implausible"] | chk["flag_inconsistent"], "patient_id"].unique()
    artifacts = pick(art_ids, 10)

    # patients used in the demo video (both held-out test patients), pinned so the video is reproducible
    early = ["p119917"] + [p for p in early if p != "p119917"]
    cats = [("Sepsis · warned ≥6h early", early), ("Sepsis · missed or late", missed),
            ("No sepsis · no alerts", stable), ("No sepsis · false alarm", fa),
            ("Real-world data artifacts", artifacts), ("Stress-test showcase", ["p018345"])]
    rows = []
    for name, ids in cats:
        for p in ids:
            on = ts.get(p, np.nan)
            desc = f"{p} · Hospital {hosp[p]} · {los[p]}h stay" + (f" · onset h{int(on)}" if not np.isnan(on) else "")
            assert p in set(te["patient_id"]), "demo patients must be held-out test patients"
            rows.append({"category": name, "patient_id": p, "label": desc})
    idx = pd.DataFrame(rows).drop_duplicates("patient_id")
    idx.to_parquet(ROOT / "app" / "demo_index.parquet", index=False)
    raw[raw["patient_id"].isin(set(idx["patient_id"]))].to_parquet(ROOT / "app" / "demo_patients.parquet", index=False)
    print(idx.groupby("category").size())


if __name__ == "__main__":
    main()
