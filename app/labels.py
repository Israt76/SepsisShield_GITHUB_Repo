"""Plain-language names for model features (used in explanations)."""
import re

BASE = {
    "HR": "Heart rate", "O2Sat": "O₂ saturation", "Temp": "Temperature", "SBP": "Systolic BP",
    "MAP": "Mean arterial pressure", "DBP": "Diastolic BP", "Resp": "Respiratory rate",
    "EtCO2": "End-tidal CO₂", "BaseExcess": "Base excess", "HCO3": "Bicarbonate", "FiO2": "FiO₂",
    "pH": "Blood pH", "PaCO2": "PaCO₂", "SaO2": "Arterial O₂ sat", "AST": "AST", "BUN": "BUN",
    "Alkalinephos": "Alkaline phosphatase", "Calcium": "Calcium", "Chloride": "Chloride",
    "Creatinine": "Creatinine", "Bilirubin_direct": "Direct bilirubin", "Glucose": "Glucose",
    "Lactate": "Lactate", "Magnesium": "Magnesium", "Phosphate": "Phosphate",
    "Potassium": "Potassium", "Bilirubin_total": "Total bilirubin", "TroponinI": "Troponin I",
    "Hct": "Hematocrit", "Hgb": "Hemoglobin", "PTT": "PTT", "WBC": "White cell count",
    "Fibrinogen": "Fibrinogen", "Platelets": "Platelets", "Age": "Age", "Gender": "Sex (male=1)",
    "Unit1": "Medical ICU", "Unit2": "Surgical ICU", "HospAdmTime": "Hours from hospital to ICU admit",
    "ICULOS": "Hours in ICU", "n_labs_cum": "Lab tests ordered so far",
    "shock_index": "Shock index (HR/SBP)", "pulse_pressure": "Pulse pressure",
    "bun_creat": "BUN/creatinine ratio", "sirs_score": "SIRS criteria met",
    "sirs_hr": "SIRS: heart rate >90", "sirs_temp": "SIRS: abnormal temperature",
    "sirs_resp": "SIRS: fast breathing", "sirs_wbc": "SIRS: abnormal WBC",
    "qsofa_partial": "qSOFA (resp + BP)", "sofa_cardio": "SOFA: MAP <70",
    "sofa_coag": "SOFA: platelets", "sofa_liver": "SOFA: bilirubin", "sofa_renal": "SOFA: creatinine",
    "sofa_resp_proxy": "SOFA: O₂ sat <92", "sofa_partial": "Partial SOFA score",
}
STAT = {"mean": "mean", "min": "min", "max": "max", "std": "variability", "diff": "change"}


def pretty(f: str) -> str:
    if f in BASE:
        return BASE[f]
    m = re.match(r"(.+)_(mean|min|max|std|diff)(\d+)$", f)
    if m:
        return f"{BASE.get(m[1], m[1])} — {STAT[m[2]]} over {m[3]}h"
    if f.endswith("_hrs_since"):
        return f"Hours since {BASE.get(f[:-10], f[:-10])} measured"
    if f.endswith("_delta"):
        return f"{BASE.get(f[:-6], f[:-6])} — change since last test"
    if f.endswith("_cummax"):
        return f"{BASE.get(f[:-7], f[:-7])} — highest so far"
    return f
