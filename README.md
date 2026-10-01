# 🛡️ SepsisShield AI: Trust-Aware Early Warning for Sepsis

**SepsisShield predicts sepsis early and runs independent input-integrity checks on the clinical data behind each
prediction, so it can warn about or withhold a prediction when those inputs look unreliable.**

Built on de-identified PhysioNet 2019 ICU data. **Research prototype only. Not a medical device and not
intended for clinical decision-making.**

---

## 30-second judge summary

| | |
|---|---|
| **Problem** | A sepsis prediction can look normal even when the clinical inputs are wrong: a °F thermometer in a °C field, a lab in the wrong units, a frozen monitor feed or an edited chart. |
| **Innovation** | SepsisShield shows **model confidence** and **input trust** as separate signals, because a model can be confident about corrupted inputs. It performs independent input-integrity checks alongside the prediction path and grades every patient-hour's inputs HIGH / REDUCED / LOW. Model disagreement may reduce trust to REDUCED, but LOW trust and prediction withholding are triggered only by evidence from the clinical inputs. |
| **Action** | HIGH → **show** the prediction · REDUCED → **warn** (verify inputs) · LOW → **withhold** until the data are checked |
| **Strongest evidence** | Held-out test set of 8,068 patients: AUROC **0.852** (95% CI 0.840–0.865). **53.9%** of septic patients alerted at least 6 h before onset. **95.4%** of alert-changing wrong decisions caused by four simulated accidental fault types were flagged or withheld (6,481 / 6,790; validation-cohort replication **94.7%**), while only **0.46%** of clean patient-hours were withheld. |
| **Also shown** | A separate **distribution-shift awareness** signal (LOW / MODERATE / HIGH): does this patient's recent input pattern differ from the training data? It is advisory only: it never changes the prediction, the input-trust state or the final decision. |
| **Limitations** | Deliberately edited inputs: only **42.5%** caught (334 / 786), and 26.5% (232 / 876) when the simulated edits stay physiologically plausible. Performance drops at an unseen hospital (AUROC **0.790 / 0.775**). Faults are simulated, not recorded hospital incidents. |
| **Try it** | Live demo: **[https://sepsisshieldapprepo-diqdknfnksrq5yyf6wkcrd.streamlit.app/](https://sepsisshieldapprepo-diqdknfnksrq5yyf6wkcrd.streamlit.app/)** · or run locally in 60 seconds (below) · demo video: linked on the Devpost project page · claim audit: [AUDIT.md](AUDIT.md) |

In this project, "clinical data" means the de-identified ICU time-series measurements in the PhysioNet 2019 Sepsis
Challenge dataset: vital signs, laboratory measurements and other patient variables recorded hour by hour.

## Live demo

**[https://sepsisshieldapprepo-diqdknfnksrq5yyf6wkcrd.streamlit.app/](https://sepsisshieldapprepo-diqdknfnksrq5yyf6wkcrd.streamlit.app/)** (Streamlit Community Cloud; first load after inactivity can take ~30 s while the app wakes). The page
opens directly on Judge Demo scenario A.

![Judge Demo Mode: edited chart, model confident, input trust LOW, prediction withheld](results/screenshots/02_prediction_withheld.png)

## Core architecture

SepsisShield separates prediction from input verification. One path estimates sepsis risk, while independent
input-integrity checks evaluate whether the clinical inputs appear reliable. A decision layer then decides whether the
prediction is shown, shown with a warning, or withheld. Model disagreement may reduce trust to REDUCED, but LOW trust and prediction withholding are triggered only by evidence from the clinical inputs.

```
                    ICU time-series data (hourly vitals, labs, demographics)
                                         |
                  +----------------------+----------------------+
                  |                                             |
          PREDICTION PATH                                  TRUST PATH
  172 causal features -> 5x LightGBM              input-integrity checks on the raw data
  -> isotonic calibration -> SHAP                 (plausibility, BP consistency, jumps,
                  |                                coordinated normalising shift, frozen feed)
       Sepsis risk + alert                                      |
       Model confidence (5-model spread)              Input trust: HIGH / REDUCED / LOW
                  |                                             |
                  +----------------------+----------------------+
                                         |
                                  DECISION LAYER
                     SHOW  /  WARN (verify inputs)  /  WITHHOLD

  Advisory, shown beside the decision (never changes it):
  DISTRIBUTION-SHIFT AWARENESS  20 key model inputs vs their training range -> LOW / MODERATE / HIGH shift
```

![Architecture](results/figures/0_architecture.png)

*Exact rule (`src/integrity.py::trust_level`):* LOW, and therefore withholding, is triggered **only by input evidence**
(physiologically implausible values, inconsistent blood pressure, a coordinated "normalising" shift). REDUCED is triggered by other input
flags or by unusual disagreement between the five models. So model confidence can lower trust to REDUCED, but it can
never cause a withhold, and it can never raise trust.

## Why SepsisShield is different

A conceptual comparison of system designs, not a claim about any specific commercial or academic system:

| | Standard sepsis predictor | Confidence-only safety | **SepsisShield** |
|---|---|---|---|
| Predicts sepsis risk | ✓ | ✓ | ✓ |
| Shows model uncertainty / confidence | – | ✓ | ✓ |
| Independently checks whether the **inputs** are believable | – | – | ✓ |
| Behaviour when corrupted inputs look plausible to the model | confident prediction | confidence may stay high | trust drops → warn or withhold |
| Explains *why* it reacted | – | – | ✓ names the check that fired |

**Why model confidence is not enough.** Ensemble agreement measures how consistent the model is with itself. It
does not show that the clinical data are correct. In Judge Demo scenarios B and C below, all five models agree
(*Confident*) while the inputs are wrong. SepsisShield therefore treats model confidence and input trust as separate
signals.

**Real-world problem.** Clinical measurements can be wrong: unit mismatches, frozen monitor feeds, typing errors and
edited chart values. An ML model receiving such inputs can still return a plausible-looking number. SepsisShield
adds an input-reliability layer *before* a prediction is trusted. It does not claim to improve patient outcomes, which
were not measured.

**Intended users (research setting):** ICU clinicians (decision *support*, not a replacement for clinical judgement),
clinical informatics teams, and hospital ML / AI monitoring teams.

**Example workflow:** ICU measurements arrive → SepsisShield estimates risk → the trust path checks the inputs → a fault
is detected (e.g. a physiologically implausible temperature) → the models may still be confident → input trust drops → the prediction
is shown with a warning or withheld → the user is asked to verify the named measurement.

## Strongest results

| Held-out test set (8,068 patients, 586 septic) | Value |
|---|---|
| AUROC | **0.852** (95% CI 0.840–0.865) |
| Sepsis patients alerted | **79.4%** (465 / 586) |
| Alerted ≥ 6 h before onset | **53.9%** (316 / 586) |
| Alert-changing wrong decisions from 4 simulated accidental fault types, flagged or withheld | **95.4%** (6,481 / 6,790); validation replication 94.7% |
| Clean patient-hours withheld | **0.46%** (1,438 / 309,270) |
| **Known limitation:** deliberately edited inputs flagged or withheld | **42.5%** (334 / 786); 26.5% (232 / 876) when the simulated edits are kept inside normal physiological ranges (sensitivity check, `results/masking_realism.json`) |
| **Known limitation:** cross-hospital AUROC (train one hospital, test the other) | **0.790 / 0.775** |

Details, confidence intervals, baselines and definitions are in [Results](#results) and
[The trust path](#the-trust-path--what-is-new).

## Judge Demo Mode

Three buttons at the top of the dashboard each load a real held-out test patient, scored live by the shipped models.
The four cards (**Sepsis risk · Model confidence · Input trust · Final decision**) update immediately. Below the cards,
judges can compare clean vs corrupted inputs and see why each decision was made:
* **Clean vs corrupted** (B and C): the same patient-hour scored on the clean and the corrupted data, side by side:
  risk, alert, model confidence, input trust and final decision. For B: 0.9%, no alert, Confident, HIGH, shown → 3.9%,
  alert, still Confident, LOW, withheld.
* **Why was this decision made?**: the rule that produced the decision, the top three SHAP risk drivers, the integrity
  reasons, and the distribution-shift state, in one compact panel.
* In *Results & limitations*, an **Evidence behind the 95.4% claim** expander gives the 6,481 / 6,790 breakdown by
  fault type, the definition, the scope (accidental faults only, with 42.5% / 26.5% for deliberate edits) and links to
  `results/abstention.json`, `src/abstention.py` and AUDIT.md.

| Scenario | Patient · hour | Model confidence | Input trust | Final decision | Shareable link |
|---|---|---|---|---|---|
| **A · Clean inputs** (normal case) | p119917 · h58 | Confident | HIGH | **SHOW PREDICTION** (this patient's first alert was at h55, 9 h before recorded onset) | `?pid=p119917&hour=58` |
| **B · Accidental data fault** (°F thermometer) | p119917 · h36 | Confident | LOW (physiologically implausible temperature) | **PREDICTION WITHHELD** (the fault would have caused a false alert, 28 h before onset and outside the useful window) | `?pid=p119917&corr=Thermometer%20reports%20°F&start=34&len=8&hour=36` |
| **C · Edited chart** | p018345 · h46 | Confident | LOW (only the coordinated-shift detector fires) | **PREDICTION WITHHELD** (the edit hid an alert the model raises on the real data: 3.5% → 2.3%) | `?pid=p018345&corr=Vitals%20overwritten%20to%20look%20normal&start=46&len=12&hour=46` |

| *Optional* **D · Unusual patient (distribution shift)** | p105030 · h40 | Confident | HIGH | **SHOW PREDICTION**, risk 0.6%, no alert. **Shift HIGH:** white cell count ~100–170, far outside the training range. This patient developed sepsis 7 h later | `?pid=p105030&hour=40` |

Move scenario B one hour later (`hour=37`) to see the third state, **VERIFY INPUTS** (trust REDUCED: the fault is
still inside the 6–12 h trend features). Scenario C shows a *caught* edit; most simulated edits are not caught (see
limitations), and later hours of C's window also contain physiologically implausible respiratory rates produced by the simulation. The expected outcome of every scenario is asserted in
`tests/test_judge_demo.py` and in the browser test, so the demo cannot drift from what the models actually do.

| Clean inputs: trust HIGH → SHOW | Accidental fault (°F): trust LOW → WITHHELD |
|---|---|
| ![](results/screenshots/01_judge_demo_normal.png) | ![](results/screenshots/03_fahrenheit_fault.png) |

## Run in 60 seconds

```bash
git clone https://github.com/Israt76/SepsisShield_GITHUB_Repo.git && cd SepsisShield_GITHUB_Repo
pip install -r requirements.txt          # exact pinned versions, Python 3.11
streamlit run app/app.py                 # opens on Judge Demo scenario A
```

No raw dataset download or retraining is required for judge mode: the five trained models, calibrator, integrity
thresholds, result files and 52 held-out demo patients ship with the repository. No API keys or credentials are needed.

```bash
python -m pytest -q tests                # 62 tests (1 needs the raw data and is skipped in judge mode)
```

Full reproduction from raw data is [below](#reproduce-everything-from-raw-data).

## Distribution-shift awareness (advisory signal)

A fourth signal, kept separate from the other three:

| Signal | Question it answers |
|---|---|
| Sepsis risk | What is the estimated clinical risk? |
| Model confidence | Do the five models agree? |
| Input trust | Do the current measurements look reliable? |
| **Distribution shift** | **Does this patient's recent input pattern differ from the training data?** |

**Method (patient-level, transparent, no extra model; `src/shift.py`).**
1. *Features:* the 20 model inputs with the highest mean LightGBM gain across the five shipped models, excluding the
   ICU-hour index (`ICULOS`), whose range reflects stay length rather than physiology or care practice. They include
   labs (WBC, creatinine, platelets, BUN, PTT, alkaline phosphatase), temperature, care-process signals (hours since
   FiO₂ / lactate / PaCO₂ / EtCO₂ were last measured, labs ordered so far, hospital-to-ICU time), age and ICU type.
2. *Reference:* each feature's 0.5th and 99.5th percentile on the **training patients only** (28,234 patients,
   1,086,027 hours; `models/shift_reference.json`).
3. *Score:* at each hour, the share of observed features outside that range, averaged over the last 6 hours
   (causal: later hours are never used). Missing values are ignored.
4. *States:* thresholds are quantiles of the same score on training patient-hours. **LOW** ≤ q95 (0.059),
   **MODERATE** ≤ q99 (0.127), **HIGH** above q99. So MODERATE means more unusual than 95% of training
   patient-hours, and HIGH more unusual than 99%.
5. The panel lists the features that triggered the state, their values and the training range. It **never** changes
   the risk, the alert, input trust or the final decision.

**What was checked (held-out data; `results/shift_evaluation.json`).**
* The thresholds transfer to held-out patients from the same hospitals: 4.6% of test hours are
  flagged MODERATE or HIGH, vs 4.8% in training.
* *Does it notice an unseen hospital?* With the reference refitted on one hospital's training patients only, hours
  flagged MODERATE or HIGH are 4.3% at the same hospital vs 7.4% at the other
  (reference A), and 4.7% vs 8.5% (reference B). This is a consistent but
  modest increase.
* *Exploratory:* on the test set, the shipped model's hour-level AUROC is 0.852 in LOW-shift hours
  (95% CI 0.838–0.865), 0.857 in MODERATE
  (0.819–0.892) and **0.749** in HIGH
  (0.621–0.862; 452 patients, patient-level
  bootstrap). In HIGH-shift hours the model also over-predicts (mean predicted 4.8% vs
  observed 3.7%). The interval is wide and overlaps, so this is suggestive, not validation.

**Limitation.** Distribution-shift awareness indicates that an input pattern differs from the training distribution;
it does not establish that a prediction is incorrect or clinically unsafe. It looks at 20 inputs one at a time, so
unusual *combinations* of individually typical values are not detected. Its thresholds flag about 5% of training hours
by construction. A data fault (e.g. a °F temperature) can also raise it, which is why the panel says "check input trust
first" when both are raised.

| LOW | MODERATE (age below training range) | HIGH (scenario D) |
|---|---|---|
| ![](results/screenshots/07_shift_low.png) | ![](results/screenshots/08_shift_moderate.png) | ![](results/screenshots/09_shift_high.png) |

---

# Technical detail

## Results

All numbers are for a **held-out test set of 8,068 patients (586 with sepsis)** that was never used for training,
tuning, calibration or threshold selection. The split is by patient and stratified by hospital and outcome. 95%
confidence intervals come from a patient-level bootstrap (500 resamples): patients are resampled with all their hours,
because hours from the same patient are correlated. Treating hours as independent would give an AUROC interval of about
0.848–0.857, roughly 2.7× too narrow.

### Prediction

| Metric | Value (95% CI) |
|---|---|
| Hour-level AUROC | **0.852** (0.840–0.865) |
| Hour-level AUPRC (prevalence 1.8%) | 0.122 (0.105–0.142) |
| Official PhysioNet 2019 utility | **0.425** (0.394–0.455) |
| Sepsis patients alerted (12 h before → 3 h after onset) | **79.4%** (75.9–82.5) |
| Sepsis patients alerted ≥ 6 h before onset | 53.9% (316 / 586) |
| Non-septic patients never alerted | 73.8% (72.8–74.9) |
| Calibration error (ECE) | 0.28 percentage points |

*Context, not a ranking:* the challenge's winning team reported a 5-fold cross-validation utility of 0.430 on the same
public data (and 0.360 on a hidden test set that included a third hospital). Evaluation protocols differ, so
the two numbers are not directly comparable.

### Baselines — same test set, same protocol

![](results/figures/8_baselines.png)

| Model | AUROC | AUPRC | Utility | Sepsis pts alerted | Non-septic never alerted |
|---|---|---|---|---|---|
| SIRS rule (≥ 2 of 4) | 0.635 | 0.029 | 0.036 | 82.1% | 26.5% |
| qSOFA-partial rule (≥ 1 of 2) | 0.576 | 0.022 | −0.054 | 86.9% | 9.7% |
| Logistic regression + isotonic | 0.751 | 0.072 | 0.284 | 76.8% | 57.4% |
| Single LightGBM | 0.852 | 0.131 | 0.428 | 83.1% | 70.2% |
| 5-model LightGBM ensemble | 0.853 | 0.130 | 0.429 | 80.9% | 72.7% |
| **Ensemble + isotonic calibration (SepsisShield model)** | **0.852** | 0.122 | 0.425 | 79.4% | 73.8% |

All rows use the same 8,068 test patients, the same metrics and validation-only threshold selection. The clinical rules
use their standard fixed cut-offs, with no tuning. Logistic regression uses median imputation and standardisation fitted
on training data, trained on a random 400,000-hour subsample of the training set for speed.

Gradient boosting is the step that matters for accuracy. The ensemble and calibration do **not** improve discrimination.
They exist for the trust design: calibrated probabilities that can be read at face value, and five models whose
disagreement is a measurable signal of model uncertainty. We report this plainly rather than claiming gains the data do
not show.

### Threshold sensitivity

![](results/figures/9_threshold.png)

The alert threshold (3.0% calibrated risk) was chosen on validation data. Test utility is flat between 1.5% and 3.5%
(0.424–0.435); the best test threshold (2.0%) was *not* the one used. The threshold is a policy choice between
sensitivity and alert burden, and the dashboard shows it.

### External validation — an unseen hospital

![](results/figures/3_cross_hospital.png)

| Train → test | Internal AUROC | **External AUROC** | Internal utility | External utility | External sens. / spec. |
|---|---|---|---|---|---|
| Hospital A → B | 0.837 | **0.790** | 0.465 | 0.295 | 62% / 77% |
| Hospital B → A | 0.861 | **0.775** | 0.390 | 0.313 | 91% / 29% |

The table above is **zero-shot** external validation: no data from the destination hospital were used.

**Local recalibration (a separate retrospective experiment, not zero-shot, not deployment evidence).** Here we *do* use
destination-hospital **outcome labels**: the isotonic calibrator and alert threshold are refitted on a labelled 20% sample
of the destination hospital's patients (the model itself is unchanged), then evaluated on the remaining 80%:

| Transfer | ECE as-is → recalibrated | Utility as-is → recalibrated | Specificity as-is → recalibrated |
|---|---|---|---|
| A → B | 0.54 → **0.11** pp | 0.299 → 0.299 | 77% → 80% |
| B → A | 0.83 → **0.20** pp | 0.316 → 0.338 | 29% → 38% |

Recalibration repairs calibration but cannot restore discrimination (AUROC is unchanged, since recalibration is
monotone). Any new site would need local recalibration *and* validation.

### Subgroup audit

![](results/figures/5_subgroups.png)

Every subgroup is shown with its patient count, number of sepsis cases and a bootstrap interval, so small-sample noise
is not mistaken for a fairness finding.

| Subgroup | Patients | Septic (prevalence) | AUROC (95% CI) | Sepsis patients alerted (95% CI) |
|---|---|---|---|---|
| Hospital A | 4,068 | 358 (8.8%) | 0.830 (0.811–0.848) | 80% (77–85) |
| Hospital B | 4,000 | 228 (5.7%) | 0.874 (0.856–0.893) | 78% (73–83) |
| Age < 45 | 1,288 | 89 (6.9%) | 0.858 (0.831–0.884) | 83% (75–92) |
| Age 45–64 | 2,997 | 197 (6.6%) | 0.846 (0.820–0.869) | 75% (68–81) |
| Age 65–79 | 2,745 | 218 (7.9%) | 0.861 (0.840–0.880) | 84% (79–89) |
| Age 80+ | 1,038 | 82 (7.9%) | 0.833 (0.795–0.867) | 73% (63–83) |
| Female | 3,517 | 228 (6.5%) | 0.857 (0.835–0.879) | 79% (74–84) |
| Male | 4,551 | 358 (7.9%) | 0.848 (0.831–0.865) | 80% (76–84) |
| Medical ICU | 2,406 | 195 (8.1%) | 0.848 (0.828–0.868) | 82% (76–86) |
| Surgical ICU | 2,510 | 126 (5.0%) | 0.861 (0.832–0.892) | 73% (65–79) |
| ICU unknown | 3,152 | 265 (8.4%) | 0.840 (0.820–0.859) | 81% (76–85) |

AUROC ranges from 0.830 to 0.874; within each attribute the intervals overlap, except Hospital A vs B. Sensitivity is
lower for patients aged 80+ (73%, CI 63–83%, 82 septic patients) and surgical-ICU patients (73%, CI 65–79%, 126 septic) than for ages 65–79 (84%, CI 79–89%). These are gaps to monitor, with wide intervals.

---

## The trust path — what is new

### Input-integrity layer (`src/integrity.py`)

It runs on raw inputs, independently of the model. **Every threshold is fitted on clean training patients; no corruption
or attack data is used to fit any threshold.** (The detector *design* was refined after inspecting test-cohort benchmark
results. The validation-cohort replication below checks that this did not inflate the results.)

| Check | Severity | Catches |
|---|---|---|
| Physiological plausibility | LOW | unit errors, sensor faults (Temp 98 "°C", Creatinine 88 "mg/dL") |
| BP consistency (DBP ≥ SBP, MAP > SBP) | LOW | inconsistent combinations |
| Coordinated normalising shift: summed, clipped z-score of HR/Resp/Temp falling and SBP/MAP rising vs the patient's own baseline; threshold at a 0.25% clean false-flag budget | LOW | overwritten or manipulated charts |
| Abrupt jump > 99.95th percentile of clean hourly changes | REDUCED | spikes (could be real deterioration, so never a veto) |
| Frozen feed (core vitals identical ≥ 8 h) | REDUCED | stalled interfaces, copied-forward values |
| Device discordance; calcium unit pattern | REDUCED | cuff vs arterial line; ionised Ca in the total-Ca field |
| Ensemble disagreement > 99th pct of validation (log-odds) | REDUCED | inputs unlike training data |
| Recent hard flag (≤ 6 h) | REDUCED | corrupted values still inside rolling features |

### Trust-aware abstention

| Input trust | System behaviour |
|---|---|
| HIGH | prediction shown normally |
| REDUCED | prediction shown with "verify inputs" |
| LOW | **prediction withheld: "requires data verification"** (research score kept for audit only) |

**Model confidence ≠ input trust.** The five models can agree perfectly about corrupted inputs, as in Judge Demo scenarios B and C. The dashboard shows the two signals side by side, and `tests/test_integrity_and_metrics.py` checks that the
input-integrity flags are computed from the inputs alone, and that corrupting an input lowers trust even when the models
agree perfectly.

### Does the shield do anything? Integrity ablation under six input failures

**Setup.** 3,086 held-out test patients (all 586 septic + 2,500 random non-septic). Each failure is injected into a
10-hour window of the **raw** data and the full pipeline is re-run. For septic patients the window is placed in the
decisive period, 12 h before to 3 h after onset (worst case); for non-septic patients it is placed at random.

**What counts.** An *alert-changing fault* is a patient-hour inside the corruption window where the fault **flipped the
alert decision and the new decision is wrong**:
* a **suppressed alert**: clean data → alert, corrupted data → no alert, inside a septic patient's useful window
  (12 h before to 3 h after onset); or
* a **spurious alert**: clean data → no alert, corrupted data → alert, *outside* any useful window (a non-septic patient,
  or a septic patient far from onset).

New alerts that a fault happens to create inside a septic patient's useful window are **not** counted, because the
decision they produce is not wrong (2,062 such hours for accidental faults, reported separately in
`results/abstention.json`). **Without the integrity layer, every alert-changing fault reaches the clinician as a
normal-looking prediction.** With it:

![](results/figures/10_abstention.png)

| Scenario | Alert-changing faults | Withheld | Flagged | Silent | **Flagged or withheld** |
|---|---|---|---|---|---|
| Thermometer reports °F | 4,329 | 1,609 | 2,708 | 12 | **99.7%** |
| Lab unit mix-up | 697 | 200 | 471 | 26 | **96.3%** |
| Monitor artifact | 1,255 | 951 | 304 | 0 | **100%** |
| Frozen feed | 509 | 6 | 232 | 271 | **46.8%** |
| **All accidental faults** | **6,790** | **2,766** | **3,715** | **309** | **95.4%** |
| **Deliberately edited inputs** | **786** | 66 | 268 | 452 | **42.5%** (26.5% if edits stay plausible) |
| Ordinary noise (control) | 87 | 2 | 9 | 76 | 12.6% |

**Read these numbers with their scope.** The 95.4% applies to the four *accidental* fault types we simulated. It does not
apply to deliberate edits (42.5%; 26.5% when the edits stay physiologically plausible) or to failure types we did not simulate.

* **By direction (accidental faults):** spurious alerts 98.3% (6,210 / 6,317); suppressed alerts 57.3% (271 / 473), limited
  mainly by frozen feeds, which by definition can only be recognised after 8 identical hours.
* **Attributable coverage:** counting only flags that the fault itself caused (the same hour was HIGH trust on clean data),
  coverage is 89.4% for accidental faults and 34.5% for deliberate edits.
* **Validation-cohort replication:** the integrity layer's *thresholds* were always fitted on training patients, but its
  *design* (severity tiers, the coordinated-shift detector) was revised after inspecting benchmark results on the test
  cohort. To check that this did not inflate the results, the full benchmark was re-run on validation patients, who
  played no part in any integrity design decision (1,543 patients: all septic plus 1,250 random non-septic, mirroring
  the test benchmark's sampling): accidental faults **94.7%** (3,313 / 3,498), deliberate edits **48.0%** (212 / 442).
  Clean predictions withheld across all 4,034 validation patients: 0.48%.
* **Patient level:** every septic patient whose alert an accidental fault suppressed received at least one trust warning
  (42 / 42), as did every non-septic patient given a new false alarm (773 / 773).

**Cost on clean data** (all 8,068 held-out test patients, no corruption):
* **0.46% of patient-hours withheld** (1,438 / 309,270) and 4.7% flagged.
* 12% of patients have at least one withheld hour, typically a single hour (median 1).
* Of the clean withholds, 48% are triggered by values that are physiologically implausible in the original records (withholding
  is correct there), and 52% by the coordinated-shift detector firing on real patients (likely false flags, e.g. rapid
  improvement after treatment).
* Of 4,846 correct sepsis-alert hours, 38 (0.8%) were withheld; 464 of the 465 detected sepsis patients still received at
  least one shown alert.

**Flagged in the original data:** 2,548 patient-hours with physiologically implausible values in the *original*
PhysioNet files (e.g. FiO₂ 4000, respiratory rate 1) and 9,884 calcium values clustered at 1.0–1.9 (median 1.17),
consistent with a possible unit mismatch (ionised calcium in mmol/L entered into a
total-calcium (mg/dL) field.

### Principal limitation: deliberate manipulation

Detection was substantially weaker for deliberately edited inputs: 42.5% of alert-changing edits were flagged or withheld
(334 / 786; 48.0% on the validation cohort). The simulated edit subtracts fixed amounts, which sometimes pushes values
outside physiology (e.g. respiratory rate below 4), and those implausible values help detection. A sensitivity check
that keeps the edited vitals inside normal adult ranges (HR ≥ 60, Resp ≥ 12, Temp ≥ 36 °C, SBP ≤ 160, MAP ≤ 110)
lowers coverage to **26.5% (232 / 876)** (`tools/masking_realism_check.py`, `results/masking_realism.json`). The
reported 42.5% is therefore an optimistic figure for realistic edits. The layer sees a jump when an edit begins; a
careful, gradual falsification that stays inside normal physiology is largely undetectable from vitals alone.
Concrete next steps: (1) **cross-signal consistency**, predicting each vital from the others and from labs and flagging
residuals (an overwritten heart rate that no longer matches lactate, WBC and respiratory trends); (2) **provenance
signals** from the EHR audit trail (who changed a value, when, and whether it was device-captured or manually entered),
which is the strongest defence in practice; (3) **conformal risk bounds** so abstention thresholds carry a
distribution-free guarantee.

---

## How the prediction path works

* **Data:** [PhysioNet/CinC Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/): 40,336 ICU patients,
  1,552,210 patient-hours, 2,932 sepsis cases (Sepsis-3 based labels) from two US hospital systems.
  See [DATA_CARD.md](DATA_CARD.md).
* **Features (172, causal):** last observed value; hours since each variable was measured (informative missingness);
  6 h / 12 h rolling statistics and changes; lab trajectories; SIRS, qSOFA- and SOFA-style composites; shock index.
  `tests/test_no_leakage.py` recomputes features on every truncated history and perturbs future values to prove
  no look-ahead.
* **Model:** LightGBM × 5 seeds, hyper-parameters chosen on validation only (`src/tune.py`,
  `results/logs/log_tune.txt`); isotonic calibration and utility-optimal threshold fitted on validation patients
  (`tests/test_isolation.py` re-derives both from the shipped validation predictions). See [MODEL_CARD.md](MODEL_CARD.md).
* **Explanations:** exact TreeSHAP contributions (LightGBM `pred_contrib`), averaged over the ensemble.

<details><summary>Further validation figures: calibration, lead time, feature ablation, global drivers</summary>

![](results/figures/1_calibration.png) ![](results/figures/2_lead_time.png)
![](results/figures/4_ablation.png) ![](results/figures/7_shap_global.png)

* The raw ensemble was already close to calibrated (ECE 0.32 pp); isotonic calibration gives 0.28 pp.
* Feature ablation (single model): labs, informative missingness and temporal trends each add signal (utility 0.388 →
  0.439); clinical composites did not (0.439 → 0.427), and they are kept for legible explanations.
* The strongest global drivers include care-process signals (time since FiO₂ / lactate were measured, number of labs
  ordered). These are predictive but hospital-specific, which is one reason for the cross-hospital drop.
</details>

---

## How SepsisShield relates to existing work

SepsisShield does not claim a better sepsis classifier. Its contribution is the **integration** of pieces that usually
live apart:

| Line of work | What it does | What SepsisShield adds |
|---|---|---|
| Sepsis prediction: PhysioNet 2019 entries [1, 11]; deployed systems such as TREWS [10] | estimate risk from EHR time series | the same task, plus an explicit decision about whether the inputs deserve trust |
| External validation of deployed models [6] and dataset shift [7] | show models degrade silently in new settings | cross-hospital validation, recalibration experiment, per-hour trust signal |
| EHR data-quality assessment [14] | offline audits of data plausibility and conformance | online, per-patient-hour checks wired into the prediction decision |
| Adversarial attacks on medical ML [8] | show small input changes can flip model outputs | a manipulation benchmark and a detector for coordinated normalising edits, with its failure rate reported |
| Uncertainty estimation (deep ensembles [9]) and selective prediction [15] | quantify model uncertainty and abstain when unsure | shows model confidence alone misses corrupted inputs; abstention is driven by input trust |

## Reproduce everything from raw data

```bash
pip install -r requirements.txt
wget -r -N -c -np -nH --cut-dirs=4 -P data/raw https://physionet.org/files/challenge-2019/1.0.0/training/
./run_all.sh            # ~55 min on 2 CPU cores; rebuilds every number, table and figure
python -m pytest -q tests
```

A from-scratch rerun reproduced `results/results.json` and `results/integrity_benchmark.json` byte for byte. Seeds, splits,
thresholds and calibrators are saved in `models/`.

```
app/            Streamlit dashboard (judge mode), Judge Demo scenarios (scenarios.py), held-out demo patients
src/shift.py    distribution-shift awareness (reference in models/shift_reference.json, evaluation in results/)
src/            data → features → training → integrity → experiments → figures
tests/          leakage, isolation, integrity, utility and pipeline tests (run in CI, plus a browser test of abstention)
models/         5 LightGBM models, calibrator, thresholds, integrity config
results/        results JSON, figures, screenshots, logs/
submission/     Devpost text, video, video script, gallery; ml_empowerment/ = competition-specific assets
tools/          screenshot, architecture and video tooling
```

## Limitations

* Retrospective data from two US hospital systems, with no prospective or real-time evaluation.
* Labels follow the challenge's Sepsis-3 based definition, shifted 6 h early; lead time is measured against that
  definition and capped at 12 h by the evaluation window.
* External (zero-shot) AUROC drops to 0.790 (train A → test B) and 0.775 (train B → test A); thresholds and calibration
  must be re-fitted locally.
* The model leans on care-process signals that differ between hospitals.
* Integrity coverage is high for three of the four accidental fault types, 47% for frozen feeds, and 42.5% for deliberate edits (26.5% when the simulated edits stay inside normal physiological ranges).
* The integrity layer's design was informed by test-cohort benchmark results; a validation-cohort replication gives similar
  numbers, but a fully independent benchmark cohort would be cleaner.
* Corruptions in the benchmark are simulated; real-world failure frequencies are unknown.
* Distribution-shift awareness indicates that an input pattern differs from the training distribution; it does not
  establish that a prediction is incorrect or clinically unsafe.
* A research prototype, not a medical device, and not validated for any clinical decision.

## References

1. Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine* 48(2):210–217 (2020). Dataset: PhysioNet, CC BY 4.0.
2. Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA* 315(8):801–810 (2016).
3. Ke G, Meng Q, Finley T, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS* (2017).
4. Lundberg SM, Erion G, Chen H, et al. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence* 2:56–67 (2020).
5. Zadrozny B, Elkan C. Transforming classifier scores into accurate multiclass probability estimates. *KDD* (2002).
6. Wong A, Otles E, Donnelly JP, et al. External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients. *JAMA Internal Medicine* 181(8):1065–1070 (2021).
7. Finlayson SG, Subbaswamy A, Singh K, et al. The Clinician and Dataset Shift in Artificial Intelligence. *NEJM* 385:283–286 (2021).
8. Finlayson SG, Bowers JD, Ito J, Zittrain JL, Beam AL, Kohane IS. Adversarial attacks on medical machine learning. *Science* 363(6433):1287–1289 (2019).
9. Lakshminarayanan B, Pritzel A, Blundell C. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS* (2017).
10. Adams R, Henry KE, Sridharan A, et al. Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis. *Nature Medicine* 28:1455–1460 (2022). doi:10.1038/s41591-022-01894-0
11. Morrill J, Kormilitzin A, Nevado-Holgado A, et al. The Signature-Based Model for Early Detection of Sepsis from Electronic Health Records in the Intensive Care Unit. *Computing in Cardiology* (2019).
12. Bone RC, Balk RA, Cerra FB, et al. Definitions for sepsis and organ failure and guidelines for the use of innovative therapies in sepsis. *Chest* 101(6):1644–1655 (1992).
13. Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of Clinical Criteria for Sepsis (qSOFA). *JAMA* 315(8):762–774 (2016).
14. Kahn MG, Callahan TJ, Barnard J, et al. A Harmonized Data Quality Assessment Terminology and Framework for the Secondary Use of Electronic Health Record Data. *eGEMs* 4(1):1244 (2016). doi:10.13063/2327-9214.1244
15. Geifman Y, El-Yaniv R. Selective Classification for Deep Neural Networks. *NeurIPS* (2017).

## Research and safety disclaimer

**Research prototype only. Not a medical device and not intended for clinical decision-making.**

SepsisShield is a research prototype built on retrospective, de-identified data from two US hospital systems. It is
not a medical device, has not been prospectively or clinically validated, and must not be used for patient care. The
trust layer reduces some failure modes in a simulated benchmark; it does not detect all bad data and does not prevent
all wrong predictions.

## Licence

Code: MIT. Data: PhysioNet/CinC Challenge 2019, Creative Commons Attribution 4.0. The data are not redistributed here
except for 52 de-identified held-out demo patients used by the dashboard.
