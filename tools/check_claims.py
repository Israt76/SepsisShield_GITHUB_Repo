"""Check every headline number in README / DEVPOST / MODEL_CARD / video script against the result artifacts."""
import json, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
DOCS = Path(__file__).resolve().parents[1]
J = lambda n: json.load(open(ROOT / "results" / n))
R, E, A, V, IB = J("results.json"), J("experiments2.json"), J("abstention.json"), J("abstention_val.json"), J("integrity_benchmark.json")
SG = {(r["attribute"], r["group"]): r for r in J("subgroups_ci.json")}
m, ci = R["main"]["calibrated"], E["bootstrap_ci"]["sepsisshield"]
acc, mk, cf = A["dfc_accidental"], A["dfc_masking"]["all"], A["clean_cost_full_test"]
S = {s["corruption"]: s["all"] for s in A["scenarios"]}
B = {b["model"]: b for b in E["baselines"]}
RC = {r["transfer"]: r for r in E["recalibration"]}
cov = lambda a: f"{(a['withheld'] + a['flagged']):,} / {a['n']:,}"
pct = lambda a: f"{(a['withheld'] + a['flagged']) / a['n'] * 100:.1f}%"

claims = {
    # prediction
    "test patients": f"{R['main']['n_patients']['test']:,}",
    "AUROC": f"{m['auroc']:.3f}", "utility": f"{m['utility']:.3f}",
    "AUROC CI": f"{ci['auroc'][0]:.3f}–{ci['auroc'][1]:.3f}", "utility CI": f"{ci['utility'][0]:.3f}–{ci['utility'][1]:.3f}",
    "AUPRC CI": f"{ci['auprc'][0]:.3f}–{ci['auprc'][1]:.3f}",
    "sensitivity": f"{m['patient_sensitivity']*100:.1f}%", "sens CI": f"{ci['patient_sensitivity'][0]*100:.1f}–{ci['patient_sensitivity'][1]*100:.1f}",
    "specificity": f"{m['patient_specificity']*100:.1f}%", "spec CI": f"{ci['patient_specificity'][0]*100:.1f}–{ci['patient_specificity'][1]*100:.1f}",
    ">=6h early": f"{m['pct_detected_ge6h_early']*100:.1f}%", "ECE": f"{m['ece']*100:.2f}",
    "external A->B": f"{R['cross_A2B']['auroc']:.3f}", "external B->A": f"{R['cross_B2A']['auroc']:.3f}",
    "baseline SIRS": f"{B['SIRS criteria (rule: ≥2 of 4)']['auroc']:.3f}", "baseline LR": f"{B['Logistic regression (+ isotonic)']['auroc']:.3f}",
    "baseline single": f"{B['Single LightGBM (uncalibrated)']['auroc']:.3f}",
    "recal B->A ECE": f"{RC['B→A']['as_is']['ece']*100:.2f} → {RC['B→A']['recalibrated']['ece']*100:.2f}",
    "recal B->A utility": f"{RC['B→A']['as_is']['utility']:.3f} → {RC['B→A']['recalibrated']['utility']:.3f}",
    # trust path (strict definition)
    "accidental coverage %": f"{acc['all']['coverage']*100:.1f}%", "accidental coverage n": cov(acc["all"]),
    "spurious": cov(acc["spurious"]), "suppressed": cov(acc["suppressed"]),
    "attributable": f"{acc['all']['attributable_coverage']*100:.1f}%",
    "masking %": f"{mk['coverage']*100:.1f}%", "masking n": cov(mk),
    "val accidental": f"{V['dfc_accidental']['all']['coverage']*100:.1f}% ({cov(V['dfc_accidental']['all'])})",
    "val masking": f"{V['dfc_masking']['all']['coverage']*100:.1f}% ({cov(V['dfc_masking']['all'])})",
    "°F": pct(S["temp_fahrenheit"]), "lab": pct(S["lab_unit_error"]), "frozen": pct(S["frozen_feed"]),
    "clean withheld %": f"{cf['withheld_pct']*100:.2f}%", "clean withheld n": f"{cf['withheld_hours']:,} / {cf['hours']:,}",
    "clean patients ever": f"{cf['patients_ever_withheld_pct']*100:.0f}% of patients",
    "real impossible hours": f"{IB['real_data_artifacts']['hours_implausible']:,}",
    # subgroups
    "80+ sens": f"73% (63–83)", "SICU sens": f"73% (65–79)",
}
# sanity: the subgroup strings above must equal the JSON
assert f"{SG[('age_group','80+')]['patient_sensitivity']*100:.0f}% ({SG[('age_group','80+')]['sens_ci'][0]*100:.0f}–{SG[('age_group','80+')]['sens_ci'][1]*100:.0f})" == claims["80+ sens"]
assert f"{SG[('icu','SICU')]['patient_sensitivity']*100:.0f}% ({SG[('icu','SICU')]['sens_ci'][0]*100:.0f}–{SG[('icu','SICU')]['sens_ci'][1]*100:.0f})" == claims["SICU sens"]

norm = lambda t: t.replace("**", "").replace(" / ", "/").replace(" ", "")
docs = {n: norm((DOCS / n).read_text()) for n in ["README.md", "submission/DEVPOST.md", "MODEL_CARD.md", "submission/VIDEO_SCRIPT.md"]}
bad = 0
for k, v in claims.items():
    where = [n.split("/")[-1] for n, t in docs.items() if norm(v) in t]
    bad += not where
    print(("OK  " if where else "MISS"), f"{k:26s} {v:28s} found in: {', '.join(where) or '—'}")
# stale numbers that must no longer appear anywhere
stale = ["8,439", "8,852", "95.3%", "372/864", "43.1%", "costs a life", "on par with"]
for n, t in docs.items():
    for sv in stale:
        if norm(sv) in t:
            bad += 1; print(f"STALE '{sv}' still in {n}")
print(f"\n{len(claims) - sum(1 for _ in [])} claims checked; problems: {bad}")
sys.exit(1 if bad else 0)
