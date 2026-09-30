"""Load the PhysioNet/CinC 2019 Sepsis Challenge PSV files into one parquet table.

Source: https://physionet.org/content/challenge-2019/1.0.0/ (CC BY 4.0)
training_setA = Hospital A, training_setB = Hospital B.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "hourly.parquet"

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
DEMO = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
LABEL = "SepsisLabel"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    df.insert(0, "patient_id", path.stem)
    df.insert(1, "hospital", "A" if "setA" in path.parent.name else "B")
    df.insert(2, "hour", np.arange(len(df), dtype=np.int16))
    return df


def build(limit: int | None = None) -> pd.DataFrame:
    files = sorted(RAW.glob("training_set*/*.psv"))
    if limit:
        files = files[:limit]
    with ProcessPoolExecutor() as ex:
        parts = list(ex.map(_read, files, chunksize=500))
    df = pd.concat(parts, ignore_index=True)
    num = VITALS + LABS + ["Age", "HospAdmTime"]
    df[num] = df[num].astype("float32")
    for c in ["Gender", "Unit1", "Unit2"]:
        df[c] = df[c].astype("float32")
    df["ICULOS"] = df["ICULOS"].astype("int16")
    df[LABEL] = df[LABEL].astype("int8")
    df["hospital"] = df["hospital"].astype("category")
    return df


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    df = build(limit)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    pat = df.groupby("patient_id").agg(h=("hospital", "first"), s=(LABEL, "max"))
    print(f"rows={len(df):,} patients={len(pat):,}")
    print(pat.groupby("h", observed=True)["s"].agg(["count", "sum", "mean"]))
