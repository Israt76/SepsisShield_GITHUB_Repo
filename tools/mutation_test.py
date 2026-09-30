"""Mutation test for the leakage tests: plant known look-ahead leaks and confirm tests/test_no_leakage.py catches each.

Run from the repository root:  python tools/mutation_test.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "features.py"
MUTANTS = [
    ("back-fill future values into missing hours",
     'ff = g[RAW_VARS].ffill()', 'ff = g[RAW_VARS].ffill().groupby(df["patient_id"], sort=False).bfill()'),
    ("centred rolling window",
     'r = gv[ROLL_VARS].rolling(w, min_periods=1)', 'r = gv[ROLL_VARS].rolling(w, min_periods=1, center=True)'),
    ("lab count over whole stay",
     'hs_cols["n_labs_cum"] = meas.groupby(df["patient_id"], sort=False).cumsum().astype("float32")',
     'hs_cols["n_labs_cum"] = meas.groupby(df["patient_id"], sort=False).transform("sum").astype("float32")'),
    ("lab maximum over whole stay",
     'lab[f"{c}_cummax"] = obs.groupby(df["patient_id"], sort=False).cummax().astype("float32")',
     'lab[f"{c}_cummax"] = obs.groupby(df["patient_id"], sort=False).transform("max").astype("float32")'),
    ("difference against a future value",
     'shifted = gv[ROLL_VARS].shift(w - 1)', 'shifted = gv[ROLL_VARS].shift(-(w - 1))'),
]


def run():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_no_leakage.py"], cwd=ROOT,
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines()[-1]


def main():
    backup = SRC.read_text()
    ok = True
    try:
        for name, old, new in MUTANTS:
            assert old in backup, f"pattern not found for mutant: {name}"
            SRC.write_text(backup.replace(old, new, 1))
            code, summary = run()
            caught = code != 0
            ok &= caught
            print(f"{'CAUGHT ' if caught else 'MISSED '} {name:38s} {summary}")
    finally:
        SRC.write_text(backup)
    code, summary = run()
    print(f"{'CLEAN  ' if code == 0 else 'BROKEN '} {'original code':38s} {summary}")
    sys.exit(0 if ok and code == 0 else 1)


if __name__ == "__main__":
    main()
