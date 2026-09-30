# SepsisShield AI — demo video script (2:28, as rendered in `submission/SepsisShield_demo.mp4`)

Three acts: **prediction → failure → trust-aware response.** The narration below is exactly what the video says, and the
captions match it. Format: 1920×1200, with the full 1920×1080 picture on top and all captions in a 120 px black band beneath it,
so no dashboard content is covered. `SepsisShield_demo_no_voice.mp4` has the same visuals without narration, if you prefer to record your own voice.

**Clinical opening (0:00–0:07.5, no narration)** — *Code-drawn animation of an ICU bedside: a patient in bed (soft focus) and a
bedside monitor with ECG, pleth and respiration traces and normal vitals; a SepsisShield decision-support panel shows risk 0.6%,
input trust HIGH, model confidence Confident. The temperature channel glitches to an impossible 98.4 "°C" (a °F value in a °C
field); the risk climbs to 3.9% and raises an ALERT (medium-priority alarm tone); SepsisShield then shows **LOW TRUST — Verify
Input Data**, strikes through the score and withholds the prediction. Values follow the real °F case for held-out patient p119917.
On-screen captions:*
> ICU patient · vital signs streaming to an early-warning model
> The thermometer starts reporting °F into a °C field
> The risk score rises — driven by a reading that cannot be real
> SepsisShield checks the inputs before trusting the prediction

**Opening** — *Problem card → title → architecture diagram*
> Most sepsis early-warning models assume their inputs are correct.
> SepsisShield tests that assumption, before asking anyone to trust its prediction.
> It has two paths. One predicts risk. The other decides whether the inputs deserve to be believed.

**Act 1 — Prediction** — *Held-out patient p119917: risk curve built hour by hour to hour 60; alert at hour 55, onset line at 64*
> Act one: prediction. A real, de-identified ICU patient from our held-out test set. At each hour, the model sees only data up to that hour.
> For two days the risk stays low. Then it rises more than tenfold, and at hour 55 SepsisShield raises an alert — 9 hours before the recorded onset of sepsis.

**Act 1 — Explanation and the two checks** — *SHAP drivers (held ~5 s), then Input trust HIGH / Model confidence Confident tiles*
> Every alert is explained. These bars show what pushed this patient's risk up, and what pulled it down.
> Beside it sit two separate checks: do the five models agree — and are the inputs themselves believable? Here, both say yes.

**Act 2 — Failure: °F thermometer** — *°F fault injected hours 34–41: score above alert line → integrity message → tiles show Withheld / Not issued / LOW*
> Act two: failure. A thermometer starts reporting Fahrenheit into a Celsius field.
> The model's score jumps above the alert line. A false alarm.
> But the integrity layer recognises an impossible temperature. Instead of an alert, the prediction is withheld until the data are verified.

**Act 2 — Failure: edited chart (second held-out patient)** — *Held-out patient p018345: clean risk 15.5% at hour 57 → edited chart 3.9% → at hour 48 the alert is not issued*
> The silent failure is worse. On a second patient, the real data show risk climbing to 15%.
> Now the chart is edited to make the patient look stable.
> The model is fooled. Risk falls to 4%, and for four hours the alert disappears.

**Act 3 — Trust-aware response** — *Model confidence 'Confident' beside Input trust 'LOW' → integrity messages → prediction Withheld*
> Act three: the trust-aware response. The five models still agree with each other, so model confidence alone would miss this.
> But the integrity layer sees several vital signs normalising at once.
> Instead of a silent all-clear, the prediction is withheld, and the data are flagged for verification.

**Act 3 — Evidence** — *Evidence card, rows revealed as narrated: 95.4% (6,481 / 6,790; validation 94.7%) · 0.46% clean withheld (1,438 / 309,270) · 42.5% deliberate (334 / 786), with the definition of an alert-changing fault*
> In our corruption benchmark, the trust layer flagged or withheld 95% of alert-changing accidental data faults (6,481 / 6,790),
> while withholding only 0.46% of clean predictions.
> Detection was substantially weaker for deliberately edited inputs — 42.5% (334 / 786). That is our main limitation, and we report it.

**Limitation** — *Cross-hospital figure (zero-shot AUROC 0.790 / 0.775)*
> Accuracy also drops at a hospital the model never trained on. Which is exactly when trust signals matter.

**Close — three numbers** — *Three numbers: 0.852 AUROC · 79% of sepsis patients alerted · 95% of alert-changing accidental data faults flagged or withheld (6,481 / 6,790); beneath: deliberately edited inputs 42.5% (334 / 786) · clean predictions withheld 0.46%*
> SepsisShield — Predict early. Explain clearly. Know when not to trust the model.

Every number in the narration comes from `results/*.json` (checked by `tools/check_claims.py`); both patients are held-out test patients.