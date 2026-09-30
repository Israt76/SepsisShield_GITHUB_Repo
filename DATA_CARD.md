# Data card — PhysioNet/CinC Challenge 2019 (as used by SepsisShield)

| | |
|---|---|
| **Source** | https://physionet.org/content/challenge-2019/1.0.0/ (training sets A and B) |
| **Licence** | Creative Commons Attribution 4.0 International |
| **Citation** | Reyna MA, et al. *Critical Care Medicine* 48(2):210–217 (2020); dataset DOI via PhysioNet |
| **Population** | ICU patients from two US hospital systems (A: 20,336 patients; B: 20,000 patients) |
| **Size** | 40,336 patients, 1,552,210 patient-hours |
| **Outcome** | `SepsisLabel` (Sepsis-3 based; set to 1 from 6 h before clinical onset). 2,932 septic patients (A 1,790 = 8.8%; B 1,142 = 5.7%) |
| **Variables** | 8 vitals (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2), 26 labs, Age, Gender, Unit1/Unit2, HospAdmTime, ICULOS |
| **De-identification** | Performed by the data providers; no direct identifiers |

## How it was obtained here
PhysioNet was not reachable from the build environment, so the files were taken from a public GitHub mirror and
checked against the official release: 40,336 files, with identical patient and sepsis counts per hospital. Judges can
download the originals with the `wget` command in the README.

## Processing
* Hourly rows kept as-is, with no imputation at the data stage. Forward-filling and missingness features are computed causally in
  `src/features.py`.
* Split: patient-level 70 / 10 / 20 (train / validation / test), stratified by hospital × outcome, seed 0.
* Cross-hospital experiments train on one hospital and test on the other.

## Known data-quality issues (found by SepsisShield's integrity layer on the original files)
* 2,548 patient-hours with physiologically impossible values (e.g. FiO₂ up to 4000, respiratory rate 1, MAP/DBP 300).
* 1,262 hours with impossible blood-pressure combinations (DBP ≥ SBP or MAP > SBP).
* 9,884 calcium values clustered at 1.0–1.9 (median 1.17), consistent with ionised calcium (mmol/L) in the total-calcium (mg/dL) field.
* Heavy, informative missingness: most labs are absent in most hours.

## Suitability and limits
* Suitable for retrospective method development and benchmarking.
* Not representative of all hospitals, countries or care settings; the hidden challenge test set included a third hospital where
  published models degraded sharply.
* Labels are algorithmic (Sepsis-3 criteria from EHR data), not chart-reviewed diagnoses.

## Redistribution in this repository
Only 52 held-out *test* patients (`app/demo_patients.parquet`) are included, so the dashboard runs without downloading the dataset.
They are attributed under CC BY 4.0.
