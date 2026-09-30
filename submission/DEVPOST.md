# SepsisShield AI — Devpost submission text

**Project name:** SepsisShield AI: Trust-Aware Early Warning for Sepsis

**Tagline (≤ 200 chars):**
A sepsis early-warning prototype that checks whether its own inputs can be trusted, and withholds its prediction when they can't.

**Track:** 02 · research prototype built on de-identified public data (not a medical device)

---

## Inspiration

Sepsis early-warning models are usually judged on how well they predict. Almost none are judged on what they do when
their inputs are wrong. Yet ICU data go wrong all the time: a thermometer reports Fahrenheit into a Celsius field, an
external lab sends SI units, a monitor feed freezes and copies values forward, a chart is edited. A standard model turns
each of these into a normal-looking prediction, with no sign that anything is off.

Our question: **can an early-warning system test whether its inputs deserve trust before asking anyone to trust its
prediction?**

## What it does

SepsisShield has two independent paths from the same raw ICU data:

* **Prediction path:** 172 causal features → 5-model LightGBM ensemble → isotonic calibration → hourly sepsis risk,
  an alert, and a SHAP explanation.
* **Trust path (the novel part):** an input-integrity layer that runs on the raw inputs, independent of the model. It
  checks physiological plausibility, blood-pressure consistency, abrupt jumps, frozen feeds, and coordinated
  "normalising" shifts that look like a chart being edited. It grades every patient-hour's inputs **HIGH / REDUCED / LOW**.

The two paths meet in a **trust-aware decision**:

| Input trust | What the system does |
|---|---|
| HIGH | shows the prediction |
| REDUCED | shows it with "verify inputs" |
| LOW | **withholds the prediction and requests data verification** |

Model confidence (do the five models agree?) and input trust (are the inputs believable?) are shown side by side,
because a model can be perfectly confident about corrupted inputs.

The Streamlit dashboard replays real held-out patients hour by hour and includes a stress-test mode that injects
realistic input faults live.

## How we built it

* **Data:** PhysioNet/CinC Challenge 2019. 40,336 ICU patients and 2,932 sepsis cases from two hospital systems.
* **Rigor:** patient-level train/validation/test split. The test set (8,068 patients) was never used for tuning,
  calibration or threshold selection. 29 automated tests cover no look-ahead in features, split isolation, calibrator
  and threshold re-derivation from validation data only, integrity behaviour, and the official utility metric; they run
  in GitHub Actions.
* **Integrity layer:** every threshold is fitted on clean training data only; no corruption or attack data is used to fit
  it. Because we refined the detector design after inspecting test-cohort benchmarks, we replicated the whole benchmark
  on the validation cohort, which played no part in that design (94.7% vs 95.4%).
* **Judge mode:** `pip install -r requirements.txt` → `streamlit run app/app.py`. Pinned dependencies, and the models and
  52 held-out demo patients ship with the repo. `./run_all.sh` rebuilds every number from the raw data; a from-scratch
  rerun reproduced the results byte for byte.
* **Stack:** Python · pandas · LightGBM · scikit-learn · Plotly · Streamlit · pytest.

## Results (held-out test set, 95% patient-bootstrap CIs)

**Prediction**
* AUROC **0.852** (0.840–0.865), PhysioNet utility **0.425** (0.394–0.455).
* **79%** of sepsis patients alerted (76–83%); 54% at least 6 hours before onset. Calibration error 0.28 percentage points.
* Baselines on the same test set: SIRS rule AUROC 0.635, logistic regression 0.751, single LightGBM 0.852. The
  ensemble and calibration don't add accuracy; they provide calibrated probabilities and a model-uncertainty signal.
* **External validation (zero-shot):** trained on one hospital and tested on the other, with no data from the new
  hospital, AUROC is 0.790 and 0.775. In a separate experiment that *does* use outcome labels from 20% of the new
  hospital's patients to refit only the calibrator and threshold, calibration improves (error 0.83 → 0.20 pp) but
  discrimination does not.
* For context only: the challenge's winning team reported 5-fold cross-validation utility of 0.430 on the same public
  data. Evaluation protocols differ, so this is not a head-to-head comparison.

**Trust path: integrity ablation under six simulated input failures (3,086 held-out patients)**
* **95.4% of alert-changing accidental data faults in our corruption benchmark were flagged or withheld (6,481 / 6,790),
  while only 0.46% of clean predictions were withheld.** An alert-changing fault is a patient-hour where the fault flipped
  the alert decision to a wrong one: it suppressed a timely sepsis alert, or created an alert where none was warranted.
  Without the integrity layer, every one of them reaches the clinician looking normal.
* By fault: °F thermometer 99.7%, lab unit mix-up 96.3%, monitor artifact 100%, frozen feed 46.8%.
* **Detection was substantially weaker for deliberately edited inputs (334 / 786, 42.5%).** This is an important
  limitation and our main direction for future work.
* Replicated on the validation cohort, which played no part in designing the integrity layer: 94.7% accidental, 48.0%
  deliberate.
* Clean-data cost across all 8,068 test patients: 0.46% of patient-hours withheld; 12% of patients have at least one
  withheld hour, typically one; 38 of 4,846 correct sepsis-alert hours withheld.
* On the *original* dataset, the layer found 2,548 physiologically impossible patient-hours, and 9,884 calcium values
  consistent with ionised calcium entered as total calcium.

## Challenges we ran into

* **A trust layer that cries wolf is useless.** Our first version marked a genuinely deteriorating patient "do not
  trust" at sepsis onset: real deterioration produces abrupt changes, and ensemble spread grows with risk. We made jumps
  reduce trust rather than veto the score, and measured model disagreement on the log-odds scale. Clean low-trust hours
  fell from 1.2% to 0.5% with no loss of detection.
* **Honest benchmarking.** Our first benchmark let risk-raising faults look *helpful* to AUROC. We replaced it with
  decision-level metrics (alerts suppressed or created), a matched clean-data baseline, and a benign-noise control.
* **Edited charts are hard.** An edit that stays inside normal physiology is invisible in vitals alone. We report 42.5%
  coverage and set out concrete next steps: cross-signal consistency and EHR audit-trail provenance.
* **Auditing our own metric.** Our first coverage definition also counted new alerts that a fault happened to create inside
  a septic patient's useful window, which aren't wrong decisions. We tightened the definition, reported the design
  iterations, and replicated the benchmark on a cohort never used for design.

## Accomplishments we're proud of

* A trust-aware abstention policy that is measured, not just displayed: coverage on one side, cost on clean data on the other.
* A test suite and CI that make the no-leakage and isolation claims checkable by anyone in two minutes; the leakage tests were
  themselves mutation-tested (five planted leaks, all caught).
* Results reported with confidence intervals, baselines, threshold sensitivity, external validation and negative
  results: the cross-hospital drop, lower sensitivity in ages 80+ and in surgical ICUs (wide intervals), and missed
  manipulations.

## What we learned

Accuracy on the development hospital is the least informative number. At an unseen hospital, utility fell 20–37% and
the alert threshold stopped transferring. Model confidence could not tell us when the inputs were broken. Trust has to be
engineered as its own component, with its own validation.

## What's next

* Cross-signal consistency models and EHR provenance signals to catch deliberate edits.
* Conformal risk bounds so abstention carries distribution-free guarantees.
* Evaluation on further public ICU datasets to test how the integrity thresholds transfer.

## How SepsisShield addresses the judging criteria

| Criterion | Evidence |
|---|---|
| **Innovation & Impact** | Input trust as its own validated component, running beside the predictor and driving a measured abstention policy. Model confidence and input trust are shown separately, because a model can be confident about corrupted inputs. Real-world relevance: the layer found 2,548 physiologically impossible hours and a likely calcium unit error in the *original* public dataset. |
| **Technical Feasibility** | Works end to end on 40,336 real ICU stays; hourly, causal, CPU-only LightGBM; runs with two commands; 29 tests plus a browser test of the abstention behaviour in CI. |
| **Rigor & Validation** | Held-out patient-level test set; bootstrap CIs; clinical-rule and logistic-regression baselines; threshold sensitivity; zero-shot cross-hospital validation; subgroup audit with counts; integrity ablation with a benign-noise control and validation-cohort replication; mutation-tested leakage tests; limitations reported, not hidden. |
| **Presentation** | Dashboard with shareable scenario links, a 2:28 video (a 7.5 s clinical opening, then three acts), a "Judge in 60 seconds" README, architecture diagram, and model and data cards. |

## Built with

python · lightgbm · scikit-learn · pandas · plotly · streamlit · pytest · github-actions · physionet

## Disclaimer

Research prototype on de-identified public data. Not a medical device; not intended for clinical decision-making.
