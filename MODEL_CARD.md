# Model card — SepsisShield AI (research prototype)

## Overview
| | |
|---|---|
| **What it is** | Hourly sepsis-risk model (5 × LightGBM + isotonic calibration) paired with an independent input-integrity layer that grades input trust HIGH / REDUCED / LOW and drives a trust-aware abstention policy. |
| **Version** | GIBC V2 submission, September 2026 |
| **Author** | Israt Jahan Aunika |
| **Licence** | MIT (code); training data CC BY 4.0 (PhysioNet) |

## Intended use
* **Intended:** research, education and method comparison on retrospective ICU data. It demonstrates trust-aware early warning:
  prediction, explanation, calibrated uncertainty and input-integrity checking in one pipeline.
* **Out of scope:** any clinical decision, triage, diagnosis or treatment. It is not a medical device, and its outputs,
  including "withheld" decisions, must not be used for patient care. It has not been validated prospectively or at any site
  beyond the two retrospective hospital systems in the training data.

## Inputs and outputs
* **Inputs:** hourly PhysioNet-2019-format rows: 8 vital signs, 26 labs, age, sex, ICU unit, hospital-to-ICU time, ICU length of stay.
* **Outputs per patient-hour:** calibrated risk (probability that the sepsis label, defined as 6 h before Sepsis-3 onset, is positive), alert
  (risk ≥ 3.0%), SHAP contributions, model confidence (5-model spread in log-odds), input trust, decision
  (SHOW / SHOW_WITH_WARNING / WITHHELD).

## Training
* 28,234 training, 4,034 validation and 8,068 test patients (patient-level split, stratified by hospital × outcome, seed 0).
* LightGBM (lr 0.02, 31 leaves, min 500 samples per leaf, feature fraction 0.5, bagging 0.8), early stopping on validation AUROC, seeds
  11/22/33/44/55. Hyper-parameters compared on validation only.
* Isotonic calibration, alert threshold and integrity-layer thresholds were fitted on validation or training patients only (re-derived
  in `tests/test_isolation.py`).

## Performance (held-out test set, 95% patient-bootstrap CI)
| Metric | Value |
|---|---|
| AUROC | 0.852 (0.840–0.865) |
| AUPRC | 0.122 (0.105–0.142), prevalence 1.8% |
| PhysioNet 2019 utility | 0.425 (0.394–0.455) |
| Sepsis patients alerted | 79.4% (75.9–82.5) |
| Non-septic patients never alerted | 73.8% (72.8–74.9) |
| ECE | 0.28 pp |
| External AUROC (A→B / B→A) | 0.790 / 0.775 |

## Subgroups (test set)
AUROC 0.830–0.874 across hospital, age band, sex and ICU type. Patient-level sensitivity is lower for ages 80+ (73%, CI 63–83%, n septic = 82)
and surgical ICU (73%, CI 65–79%, n septic = 126). See `results/subgroups_ci.json` and README figure 5.

## Trust layer (corruption benchmark, held-out test cohort)
* **95.4% of alert-changing accidental data faults were flagged or withheld (6,481 / 6,790).** An alert-changing fault is a
  patient-hour where a simulated °F, lab-unit, monitor-artifact or frozen-feed fault flipped the alert decision to a wrong
  one. Validation-cohort replication: 94.7% (3,313 / 3,498).
* **Deliberately edited inputs: 42.5% (334 / 786), substantially weaker.** This is the principal limitation.
* Clean data: 0.46% of patient-hours withheld (1,438 / 309,270); 12% of patients have at least one withheld hour (median 1 h).
* Design caveat: thresholds were fitted on training data, but detector design was revised after inspecting test-cohort
  results; the validation replication addresses this.

## Known limitations and risks
* Distribution shift: performance and calibration degrade at an unseen hospital, and thresholds do not transfer.
* Reliance on care-process signals (measurement timing and frequency) that encode local practice.
* Label definition: challenge Sepsis-3 labels are retrospective and shifted 6 h early; lead time is capped at 12 h.
* The integrity layer's thresholds reflect the training hospitals' data; its coverage of real-world faults is estimated only from simulations.
* Alert burden: at the chosen threshold, 26% of non-septic patients receive at least one alert during their stay.

## Ethical considerations
Automated sepsis alerts can cause alert fatigue, or delay care if trusted blindly. SepsisShield's design makes the model's
limits visible (withholding, warnings, explanations), but it has not been evaluated with clinicians. Subgroup gaps should be
re-assessed on local data before any further study.
