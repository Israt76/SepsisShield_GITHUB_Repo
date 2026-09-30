import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FULL_DATA = (ROOT / "data" / "processed" / "hourly.parquet").exists()
needs_full_data = pytest.mark.skipif(not FULL_DATA, reason="full PhysioNet data not present (judge mode)")


@pytest.fixture(scope="session")
def demo_raw():
    return pd.read_parquet(ROOT / "app" / "demo_patients.parquet").sort_values(["patient_id", "hour"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def root():
    return ROOT
