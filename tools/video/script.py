"""Narration: each scene is a list of (spoken_text, caption_text). Caption None -> same as spoken."""
SCENES = [
    ("title", [
        ("Most sepsis early-warning models assume their inputs are correct.", None),
        ("SepsisShield tests that assumption, before asking anyone to trust its prediction.", None),
        ("It has two paths. One predicts risk. The other decides whether the inputs deserve to be believed.", None),
    ]),
    ("monitor", [
        ("Act one: prediction. This is a real, de-identified I C U patient from our held-out test set. At each hour, the model sees only the data up to that hour.",
         "Act one: prediction. A real, de-identified ICU patient from our held-out test set. At each hour, the model sees only data up to that hour."),
        ("For two days the risk stays low. Then it rises more than tenfold, and at hour fifty-five, SepsisShield raises an alert, nine hours before the recorded onset of sepsis.",
         "For two days the risk stays low. Then it rises more than tenfold, and at hour 55 SepsisShield raises an alert — 9 hours before the recorded onset of sepsis."),
    ]),
    ("explain", [
        ("Every alert is explained. These bars show what pushed this patient's risk up, and what pulled it down.", None),
        ("And beside it sit two separate checks: do the five models agree, and are the inputs themselves believable? Here, both say yes.",
         "Beside it sit two separate checks: do the five models agree — and are the inputs themselves believable? Here, both say yes."),
    ]),
    ("fahr", [
        ("Act two: failure. A thermometer starts reporting Fahrenheit into a Celsius field.", None),
        ("The model's score jumps above the alert line. A false alarm.", None),
        ("But the integrity layer recognises an impossible temperature. Instead of an alert, the prediction is withheld until the data are verified.", None),
    ]),
    ("mask", [
        ("The silent failure is worse. On a second patient, the real data show risk climbing to fifteen percent.",
         "The silent failure is worse. On a second patient, the real data show risk climbing to 15%."),
        ("Now the chart is edited to make the patient look stable.", None),
        ("The model is fooled. Risk falls to four percent, and for four hours the alert disappears.",
         "The model is fooled. Risk falls to 4%, and for four hours the alert disappears."),
    ]),
    ("response", [
        ("Act three: the trust-aware response. The five models still agree with each other, so model confidence alone would miss this.", None),
        ("But the integrity layer sees several vital signs normalising at once.", None),
        ("Instead of a silent all-clear, the prediction is withheld, and the data are flagged for verification.", None),
    ]),
    ("evidence", [
        ("In our corruption benchmark, the trust layer flagged or withheld ninety-five percent of the alert-changing accidental data faults,",
         "In our corruption benchmark, the trust layer flagged or withheld 95% of alert-changing accidental data faults (6,481 / 6,790),"),
        ("while withholding fewer than half a percent of clean predictions.",
         "while withholding only 0.46% of clean predictions."),
        ("Detection was much weaker for deliberately edited inputs, about forty-two percent. That is our main limitation, and we report it.",
         "Detection was substantially weaker for deliberately edited inputs — 42.5% (334 / 786). That is our main limitation, and we report it."),
    ]),
    ("limits", [
        ("Accuracy also drops at a hospital the model never trained on. Which is exactly when trust signals matter.", None),
    ]),
    ("end", [
        ("SepsisShield. Predict early. Explain clearly. Know when not to trust the model.",
         "SepsisShield — Predict early. Explain clearly. Know when not to trust the model."),
    ]),
]
