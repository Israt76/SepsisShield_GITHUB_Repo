# Devpost submission guide — SepsisShield AI

Everything to paste or upload, field by field. Check the official GIBC V2 page for any extra required questions
(eligibility, team, custom fields) and for whether the project must be built during the event.

## 1. Before you open Devpost

1. **Push to GitHub** (public repository), from the unzipped `sepsisshield/` folder:
   ```bash
   git init
   git add .
   git commit -m "SepsisShield AI: trust-aware early warning for sepsis (GIBC V2)"
   git branch -M main
   git remote add origin https://github.com/<your-username>/sepsisshield-ai.git
   git push -u origin main
   ```
   The raw PhysioNet data are excluded by `.gitignore`. The largest file is about 10 MB (the video), well under GitHub's
   100 MB limit.
2. **Confirm CI turns green:** open the repository's **Actions** tab. The `CI` workflow runs automatically on the push:
   install → 29 tests → dashboard health check → browser test of the abstention behaviour. It takes about 5–8 minutes.
   Only link the repository in Devpost once it is green. If a step fails, the log names the step; send it to me.
3. **Upload the video** to YouTube (Unlisted is fine) or Vimeo. Devpost takes a *link*, not a file.
   * File: `submission/SepsisShield_demo.mp4` (2:28, 1920×1200: full 1080p picture plus a 120 px subtitle band; opens with a 7.5 s clinical animation)
   * Title: `SepsisShield AI — Prediction + Trust (GIBC V2, Track 02)`
   * Custom thumbnail: `submission/gallery/01_hero_video_thumbnail_1920x1080.png`
   * Description: paste the tagline below plus the GitHub link.

## 2. Devpost fields

| Field | What to enter |
|---|---|
| **Project name** | SepsisShield AI: Trust-Aware Early Warning for Sepsis |
| **Elevator pitch / tagline** | A sepsis early-warning prototype that checks whether its own inputs can be trusted, and withholds its prediction when they can't. |
| **Thumbnail** | `gallery/00_devpost_thumbnail_1500x1000.png` (3:2) |
| **About the project** | Paste `DEVPOST.md` from "## Inspiration" to the end (it is Devpost Markdown) |
| **Built with** | python, lightgbm, scikit-learn, pandas, plotly, streamlit, pytest, github-actions, physionet |
| **Try it out** | GitHub repository URL (judge mode: `pip install -r requirements.txt` → `streamlit run app/app.py`) |
| **Video demo link** | the YouTube / Vimeo URL |
| **Track** | Track 02 |

## 3. Image gallery (upload in this order, with these captions)

| # | File | Caption |
|---|---|---|
| 1 | `01_hero_video_thumbnail_1920x1080.png` | Prediction + trust: an early alert on one path, a withheld prediction on the other. |
| 2 | `02_architecture.png` | Two independent paths from the same raw data: one predicts risk, one decides whether the inputs deserve to be believed. |
| 3 | `03_prediction_withheld.png` | Edited chart on a held-out patient: the five models are confident, the inputs are not believable, so the prediction is withheld. |
| 4 | `04_trust_evidence.png` | Corruption benchmark: 95.4% of alert-changing accidental data faults flagged or withheld (6,481 / 6,790); 0.46% of clean predictions withheld; deliberately edited inputs substantially weaker at 42.5% (334 / 786). |
| 5 | `05_early_warning.png` | Held-out patient: alert at hour 55, nine hours before the recorded onset of sepsis, with all inputs trusted. |
| 6 | `06_validation_dashboard.png` | Validation evidence built into the dashboard: AUROC 0.852 (95% CI 0.840–0.865), zero-shot external AUROC 0.790 / 0.775. |
| 7 | `07_abstention_by_fault.png` | Where fault-induced wrong decisions end up, by fault type; frozen feeds (46.8%) and deliberate edits (42.5%) are the weak spots. |
| 8 | `08_cross_hospital.png` | Zero-shot cross-hospital validation: performance drops at an unseen hospital, which is exactly when trust signals matter. |

## 4. Last checks before pressing Submit

- [ ] GitHub repository is public and the Actions run is green
- [ ] Video link plays in a private/incognito window
- [ ] Every mention of 95.4% says *accidental* faults *in the corruption benchmark*, with 42.5% for deliberate edits nearby
- [ ] "Research prototype, not a medical device" appears on the Devpost page
- [ ] Team members and eligibility fields completed on Devpost
- [ ] Submitted before the deadline (leave an hour of margin for upload issues)
