# Pre-submission audit

An independent re-check of every central claim before submission. Each item says what was checked, how, the result, and
what (if anything) changed. Scripts are in `tools/` and `tests/`, so anyone can re-run them.

## 1. Tests and no-leakage

| Check | Method | Result |
|---|---|---|
| Test suite from the submitted package | Unzipped the package into an empty folder, fresh virtualenv, `pip install -r requirements.txt`, `pytest` | 28 passed, 1 skipped (needs raw data) |
| Do the leakage tests have teeth? | **Mutation testing:** planted five leaks in `src/features.py` (back-fill, centred rolling window, stay-level lab count, stay-level lab max, diff against a future value) | All five caught (7 of 13 leakage tests fail each time); original code passes all 13. Re-run: `python tools/mutation_test.py` |
| Future values | Every one of the 34 raw variables perturbed after hour *t*, including filling future gaps | Features at hours ≤ *t* unchanged |
| Labels | Flip `SepsisLabel` | Features unchanged |
| Integrity layer causal | Trust flags on truncated vs full histories | Identical at shared hours |

*Changed:* the future-perturbation test now covers all raw variables (previously 8). Label-invariance and integrity-causality
tests were added (29 tests in total).

## 2. The central metric: alert-changing accidental faults flagged or withheld

**Independent recomputation.** A fresh script (not importing the experiment's classification code) re-ran the pipeline on
clean and corrupted data and reproduced the originally reported 8,439 / 8,852 = 95.3% exactly.

**Definition weakness found and fixed.** The original count included 2,062 patient-hours where a fault created a *new
alert inside a septic patient's useful window* (12 h before to 3 h after onset). That decision is not wrong, so it
should not count as a dangerous failure. The definition is now:

> An **alert-changing fault** is a patient-hour inside the corruption window where the fault flipped the alert decision
> **and the new decision is wrong**: it suppressed a timely sepsis alert, or created an alert outside any useful window.

| | Before (flip only) | **After (flip and wrong)** |
|---|---|---|
| Accidental faults flagged or withheld | 8,439 / 8,852 = 95.3% | **6,481 / 6,790 = 95.4%** |
| Deliberately edited inputs | 372 / 864 = 43.1% | **334 / 786 = 42.5%** |

**Attributable coverage.** Counting only flags caused by the fault itself (the same hour was HIGH trust on clean data)
gives 89.4% for accidental faults and 34.5% for deliberate edits. Both numbers are reported.

**Design leakage found and addressed.** Thresholds were always fitted on training data, but the detector *design* was
revised after inspecting test-cohort benchmark results. The full benchmark was therefore re-run on the 4,034 validation
patients, who played no part in any integrity design decision:

| | Test cohort | Validation cohort |
|---|---|---|
| Accidental faults flagged or withheld | 95.4% (6,481 / 6,790) | 94.7% (3,313 / 3,498) |
| Deliberately edited inputs | 42.5% (334 / 786) | 48.0% (212 / 442) |
| Clean patient-hours withheld | 0.46% | 0.48% |

**Scope wording.** Every mention now reads "95.4% of alert-changing *accidental* data faults *in our corruption
benchmark*", with the deliberate-edit result (42.5%) next to it. It never implies coverage of deliberate attacks or of
fault types that were not simulated.

## 3. Clean abstention rate

The originally reported 0.48% came from the benchmark sample, which over-represents septic patients (19% vs 7%). It was
recomputed on **all 8,068 test patients with no corruption**:

* **0.46% of patient-hours withheld** (1,438 / 309,270); 4.7% flagged.
* **12% of patients** have at least one withheld hour, typically a single hour (median 1).
* Causes of clean withholds: 48% physically impossible values or blood-pressure combinations *in the original
  records* (withholding is correct); 52% the coordinated-shift detector firing on real patients (likely false flags).
* `decision == WITHHELD` coincides exactly with `trust == LOW` (verified on every row).

## 4. Bootstrap confidence intervals

The code resamples **patients** (with all their hours), not hours. An independent reimplementation with a different
random stream gave an AUROC interval of 0.839–0.864, matching the shipped 0.840–0.865. For contrast, a naive hour-level
bootstrap gives 0.848–0.857, about 2.7× too narrow. The README now explains this.

## 5. Baselines

SIRS (0.635), qSOFA-partial (0.576), logistic regression (0.751), single LightGBM (0.852), ensemble (0.853) and the final
model (0.852) all use **the same 8,068 test patients, the same hourly rows and the same metric code**
(`experiments2.py`, one `dte` frame). Thresholds for learned models were chosen on validation data. The clinical rules
use fixed standard cut-offs, with no tuning. *Disclosed:* logistic regression was trained on a random 400,000-hour
subsample of the training set.

## 6. Threshold sensitivity: test-set leakage?

The submitted threshold (3.0%) comes from `models/config.json`. `tests/test_isolation.py` re-derives it from the shipped
validation predictions alone. The test-set sweep is reporting only, and its best threshold (2.0%) was **not** adopted.

## 7. Hospital recalibration

The zero-shot external results (AUROC 0.790 / 0.775) use no destination-hospital data. The recalibration experiment
**does use destination-hospital outcome labels** (a labelled 20% sample) to refit the calibrator and threshold only. The
README and Devpost now say this explicitly and label it as separate from zero-shot validation.

## 8. Subgroups

Patient counts, sepsis cases and patient-level prevalence were recomputed independently from `test_predictions.parquet`;
all 11 groups match `subgroups_ci.json`. The README table now includes prevalence and both intervals, generated directly
from the JSON (a hand-typed version was diffed and replaced).

## 9. Dashboard abstention behaviour

`tools/dashboard_abstention_check.py` drives the running dashboard in a real browser:

| Scenario | Expected | Observed |
|---|---|---|
| Clean, trusted inputs | shown | shown |
| °F fault inside window | withheld | withheld ("Withheld", "Not issued", LOW) |
| Edited chart, hour 48 | withheld | withheld |
| Edited chart, hour 57 | shown with warning | shown with warning |

Added to GitHub Actions.

## 10. Clean install and CI

The submitted package was installed into a new virtualenv with the pinned `requirements.txt`; the tests pass and the
dashboard health endpoint responds. The workflow YAML parses, and its steps are install → tests → dashboard health →
browser abstention test. *Not verified:* an actual GitHub Actions run, which requires the repository to be pushed.

## 11. Claims against artifacts

`tools/check_claims.py` recomputes 37 headline numbers directly from `results/*.json` and checks each appears in the
README, Devpost, model card or video script. It also fails if superseded numbers (8,439, 8,852, 95.3%, 372/864, 43.1%) or
removed phrases ("costs a life", "on par with") reappear. Result: 0 problems.

## 12. Video

Sampled every 3 s (46 frames) and reviewed. *Fixed:* a 1.3-second explanation shot that dissolved into a double
exposure; crossfades shortened; the benchmark figure (too small at video size) replaced by an evidence card that
reveals 95.4% → 0.46% → 42.5% as each is narrated, with the definition on screen; the end card now holds 3.5 s after
the narration and states the deliberate-edit result beneath the three numbers. Audio −16.6 dB mean, −1.4 dB peak.

**Clinical opening (added after the audit):** a 7.5 s code-drawn animation (`tools/video/intro.html`,
`render_intro.py`). Its numbers match the real °F case for held-out patient p119917 (temperature 36.9 → 98.4 "°C", risk 0.6% →
3.9%, alert threshold 3.0%, then LOW trust and withheld) and it is labelled on screen as an illustration.

## 13. References

Checked for existence and details; DOIs added where the metadata was confirmed (Kahn 2016 eGEMs; Adams 2022 *Nature
Medicine*).

## Known limitations that remain

* The integrity design was informed by test-cohort results; validation replication mitigates this but is not a fully
  independent third cohort.
* Deliberately edited inputs: 42.5% coverage.
* Simulated corruptions only; real-world fault frequencies are unknown.
* External AUROC drops to 0.78–0.79.
