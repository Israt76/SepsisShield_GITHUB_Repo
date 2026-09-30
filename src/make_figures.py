"""Generate result figures for the README, Devpost page and dashboard."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from labels import pretty
from metrics import onset_hour

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": SURF,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "axes.titlelocation": "left", "legend.frameon": False, "axes.axisbelow": True})


def save(fig, name):
    fig.tight_layout(rect=(0, 0.06, 1, 1) if fig.legends else None)
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


def reliability(te):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for col, lab, c in (("prob_raw", "Ensemble (raw)", ORANGE), ("prob", "After isotonic calibration", BLUE)):
        p, y = te[col].to_numpy(), te["SepsisLabel"].to_numpy()
        q = np.quantile(p, np.linspace(0, 1, 16))
        idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 14)
        mp = [p[idx == b].mean() for b in range(15)]
        my = [y[idx == b].mean() for b in range(15)]
        ax.plot(np.array(mp) * 100, np.array(my) * 100, "-o", color=c, lw=2, ms=6, label=lab,
                markeredgecolor="white", markeredgewidth=1.5)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "--", color=GRAY, lw=1, label="Perfect calibration")
    ax.set_xlabel("Predicted risk (%)"); ax.set_ylabel("Observed sepsis-label rate (%)")
    ax.set_title("1 · Calibration on held-out test set")
    ax.legend(loc="upper left")
    save(fig, "1_calibration.png")


def lead_time(te, thr):
    ts = onset_hour(te)
    d = te.assign(t_s=te["patient_id"].map(ts))
    s = d[d["t_s"].notna() & (d["prob"] >= thr) & (d["hour"] >= d["t_s"] - 12) & (d["hour"] <= d["t_s"] + 3)]
    first = s.groupby("patient_id").apply(lambda g: (g["t_s"] - g["hour"]).max())
    n_sep = ts.notna().sum()
    bins = np.arange(-3.5, 13.5, 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.hist(first, bins=bins, color=BLUE, edgecolor="white", linewidth=2)
    ax.axvline(5.5, color=INK2, ls="--", lw=1)
    ax.text(5.3, ax.get_ylim()[1] * 0.55, "≥6h early ←", color=INK2, fontsize=9, ha="right")
    ax.set_xticks([12, 10, 8, 6, 4, 2, 0, -2], ["≥12", "10", "8", "6", "4", "2", "0", "−2"])
    ax.set_xlabel("First alert: hours before clinical onset"); ax.set_ylabel("Sepsis patients")
    ax.set_title("2 · How early SepsisShield warns")
    ax.text(0.53, 0.97, f"{len(first)}/{n_sep} septic test patients alerted\n"
            f"{(first >= 6).sum()} ({(first >= 6).sum()/n_sep:.0%}) at least 6h before onset",
            transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.invert_xaxis()
    save(fig, "2_lead_time.png")


def cross(R):
    rows = [("A → B", R["cross_A2B"]), ("B → A", R["cross_B2A"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0))
    for ax, key, lab in ((axes[0], "auroc", "AUROC"), (axes[1], "utility", "Challenge utility")):
        x = np.arange(len(rows)); w = 0.36
        a = [r["internal_val"][key] for _, r in rows]; b = [r[key] for _, r in rows]
        ax.bar(x - w/2 - 0.01, a, w, color=BLUE, label="Same hospital (internal)")
        ax.bar(x + w/2 + 0.01, b, w, color=ORANGE, label="Unseen hospital (external)")
        for i in range(len(rows)):
            ax.text(x[i] - w/2 - 0.01, a[i], f"{a[i]:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
            ax.text(x[i] + w/2 + 0.01, b[i], f"{b[i]:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(x, [f"Train {r[0]}" for r in rows]); ax.set_title(lab, fontsize=11)
        ax.set_ylim(0, max(a + b) * 1.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle("3 · Cross-hospital generalization", x=0.02, ha="left", fontweight="bold", color=INK)
    save(fig, "3_cross_hospital.png")


def ablation(R):
    d = pd.DataFrame(R["ablation"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["utility"], color=BLUE, height=0.55)
    for yi, u, a in zip(y, d["utility"], d["auroc"]):
        ax.text(u + 0.005, yi, f"{u:.3f}  (AUROC {a:.3f})", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, d["config"]); ax.set_xlabel("Challenge utility (test)")
    ax.set_xlim(0, d["utility"].max() * 1.6)
    ax.set_title("4 · Ablation: what each feature group adds")
    save(fig, "4_ablation.png")


def subgroups(R):
    d = pd.DataFrame(R["subgroups"])
    d["name"] = d["attribute"].map({"hospital": "Hospital", "age_group": "Age", "sex": "Sex", "icu": "ICU"}) + ": " + d["group"]
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    y = np.arange(len(d))[::-1]
    overall = R["main"]["calibrated"]["auroc"]
    ax.axvline(overall, color=GRAY, ls="--", lw=1)
    ax.text(overall + 0.002, -0.9, f"overall {overall:.3f}", color=INK2, fontsize=8.5)
    ax.hlines(y, 0.75, d["auroc"], color=GRID, lw=1.5)
    ax.plot(d["auroc"], y, "o", color=BLUE, ms=8, markeredgecolor="white", markeredgewidth=1.5)
    for yi, a, n in zip(y, d["auroc"], d["n_patients"]):
        ax.text(a + 0.004, yi, f"{a:.3f}  (n={n:,})", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, d["name"]); ax.set_xlim(0.75, 0.95); ax.set_ylim(-1.3, len(d) - 0.5); ax.set_xlabel("AUROC (test)")
    ax.set_title("5 · Subgroup performance")
    save(fig, "5_subgroups.png")


def integrity_fig(IB):
    d = pd.DataFrame(IB["corruptions"])
    names = {"temp_fahrenheit": "Thermometer\nin °F", "lab_unit_error": "Lab unit\nmix-up",
             "sensor_artifact": "Monitor\nartifact", "frozen_feed": "Frozen\nfeed",
             "masking_attack": "Vitals overwritten\nto look normal", "measurement_noise": "Ordinary noise\n(control)"}
    d["name"] = d["corruption"].map(names)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(len(d)); w = 0.38
    ax.bar(x - w/2 - 0.01, d["patient_detect_rate"] * 100, w, color=BLUE, label="Corrupted patients flagged")
    sw = d["suppressed_warned_pct"].fillna(0) * 100
    ax.bar(x + w/2 + 0.01, sw, w, color=ORANGE, label="Hidden sepsis cases that got a trust warning")
    for i in range(len(d)):
        ax.text(x[i] - w/2 - 0.01, d["patient_detect_rate"].iloc[i] * 100 + 1, f"{d['patient_detect_rate'].iloc[i]*100:.0f}%",
                ha="center", fontsize=8, color=INK)
        n_sup = d["alerts_suppressed"].iloc[i]
        lbl = f"{sw.iloc[i]:.0f}%\n(n={n_sup})" if n_sup else "n=0"
        ax.text(x[i] + w/2 + 0.01, sw.iloc[i] + 1, lbl, ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, d["name"], fontsize=8.5); ax.set_ylim(0, 112); ax.set_ylabel("%")
    base = d["clean_same_window_flag_rate"] * 100
    ax.scatter(x - w/2 - 0.01, base, marker="_", s=500, color=INK, linewidths=2, zorder=3,
               label="Same hours, clean data (false-flag baseline)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5, ncol=2)
    ax.set_title("6 · Integrity layer vs. injected input failures")
    save(fig, "6_integrity.png")


def shap_global(te):
    cfg = json.load(open(ROOT / "models" / "config.json"))
    feats = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    ids = set(te["patient_id"].unique())
    X = feats[feats["patient_id"].isin(ids)].sample(20000, random_state=0)[cfg["features"]]
    m = lgb.Booster(model_file=str(ROOT / "models" / f"lgb_seed{cfg['seeds'][0]}.txt"))
    c = m.predict(X, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(c).mean(0), index=cfg["features"]).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    y = np.arange(len(imp))[::-1]
    ax.barh(y, imp.values, color=BLUE, height=0.6)
    ax.set_yticks(y, [pretty(f) for f in imp.index], fontsize=8.5)
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title("7 · Global drivers of predicted risk")
    save(fig, "7_shap_global.png")
    return imp


if __name__ == "__main__":
    R = json.load(open(RES / "results.json"))
    thr = json.load(open(ROOT / "models" / "config.json"))["threshold"]
    te = pd.read_parquet(RES / "test_predictions.parquet")
    reliability(te); lead_time(te, thr); cross(R); ablation(R); subgroups(R)
    if (RES / "integrity_benchmark.json").exists():
        integrity_fig(json.load(open(RES / "integrity_benchmark.json")))
    imp = shap_global(te)
    print(imp.round(4).to_string())


# ---------------------------------------------------------------- figures for experiments2 / abstention
def baselines_fig(E):
    d = pd.DataFrame(E["baselines"])
    short = {"SIRS criteria (rule: ≥2 of 4)": "SIRS rule (≥2 of 4)", "qSOFA-partial (rule: ≥1 of 2)": "qSOFA-partial rule",
             "Logistic regression (+ isotonic)": "Logistic regression", "Single LightGBM (uncalibrated)": "Single LightGBM",
             "5-model LightGBM ensemble (uncalibrated)": "5-model ensemble",
             "Ensemble + isotonic calibration (SepsisShield model)": "Ensemble + calibration\n(SepsisShield model)"}
    d["name"] = d["model"].map(short)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    y = np.arange(len(d))[::-1]
    for ax, key, lab, lo in ((axes[0], "auroc", "AUROC", 0.5), (axes[1], "utility", "Challenge utility", -0.1)):
        cols = [BLUE if "SepsisShield" in m else "#9ec5f4" for m in d["model"]]
        ax.barh(y, d[key] - (lo if key == "auroc" else 0), left=lo if key == "auroc" else 0, color=cols, height=0.6)
        for yi, v in zip(y, d[key]):
            ax.text(max(v, 0) + 0.008, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
        ax.set_title(lab, fontsize=11)
        ax.set_xlim(lo, max(d[key]) * 1.18)
        if key == "utility":
            ax.axvline(0, color=GRAY, lw=1)
    axes[0].set_yticks(y, d["name"], fontsize=9)
    fig.suptitle("8 · Baselines on the same held-out test set", x=0.02, ha="left", fontweight="bold", color=INK)
    save(fig, "8_baselines.png")


def threshold_fig(E):
    s = pd.DataFrame(E["threshold_sweep"]["rows"]); s = s[s["threshold"] <= 0.12]
    chosen = E["threshold_sweep"]["chosen"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    ax.plot(s["threshold"] * 100, s["utility_val"], "-", color=GRAY, lw=2, label="Validation (used to choose)")
    ax.plot(s["threshold"] * 100, s["utility_test"], "-", color=BLUE, lw=2, label="Test (held out)")
    ax.axvline(chosen * 100, color=INK2, ls="--", lw=1)
    ax.text(chosen * 100 + 0.15, s["utility_val"].max() * 1.005, f"chosen on validation: {chosen*100:.1f}%", color=INK2, fontsize=8.5)
    ax.set_xlabel("Alert threshold (calibrated risk, %)"); ax.set_ylabel("Challenge utility")
    ax.set_title("Utility is flat around the chosen threshold", fontsize=10.5); ax.legend(fontsize=8.5, loc="lower right")
    ax = axes[1]
    ax.plot(s["threshold"] * 100, s["patient_sensitivity"] * 100, "-", color=BLUE, lw=2, label="Sepsis patients alerted")
    ax.plot(s["threshold"] * 100, s["patient_specificity"] * 100, "-", color=ORANGE, lw=2, label="Non-septic patients never alerted")
    ax.axvline(chosen * 100, color=INK2, ls="--", lw=1)
    ax.set_xlabel("Alert threshold (calibrated risk, %)"); ax.set_ylabel("% of patients (test)")
    ax.set_title("The sensitivity / alert-burden trade-off", fontsize=10.5); ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle("9 · Threshold sensitivity", x=0.02, ha="left", fontweight="bold", color=INK)
    save(fig, "9_threshold.png")


def abstention_fig(A):
    names = {"temp_fahrenheit": "Thermometer in °F", "lab_unit_error": "Lab unit mix-up", "sensor_artifact": "Monitor artifact",
             "frozen_feed": "Frozen feed", "masking_attack": "Chart overwritten\n(deliberate)", "measurement_noise": "Ordinary noise\n(control)"}
    rows = [(names[s["corruption"]], s["all"]) for s in A["scenarios"]]
    acc = A["dfc_accidental"]["all"]
    rows.insert(4, ("All accidental faults", acc))
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    y = np.arange(len(rows))[::-1]
    for yi, (nm, r) in zip(y, rows):
        n = r["n"]; w, f, s = r["withheld"] / n * 100, r["flagged"] / n * 100, r["silent"] / n * 100
        ax.barh(yi, w, color=RED, height=0.62, label="Withheld (LOW trust)" if yi == y[0] else None)
        ax.barh(yi, f, left=w + 0.4, color="#fab219", height=0.62, label="Shown with warning (REDUCED)" if yi == y[0] else None)
        ax.barh(yi, s, left=w + f + 0.8, color="#d6d5d0", height=0.62, label="Silent (HIGH trust)" if yi == y[0] else None)
        ax.text(101.5, yi, f"{(w+f):.0f}% covered  ({r['withheld']+r['flagged']:,}/{n:,})", va="center", fontsize=8.5,
                color=INK, fontweight="bold" if nm == "All accidental faults" else "normal")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 100); ax.set_xlabel("% of alert-changing faults (patient-hours where the fault flipped the alert decision to a wrong one)")
    ax.axhline(y[4] - 0.5, color=GRID, lw=1); ax.axhline(y[4] + 0.5, color=GRID, lw=1)
    c = A.get("clean_cost_full_test", A["clean_cost"])
    ax.set_title("10 · Trust-aware abstention: where do fault-induced wrong decisions end up?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3, fontsize=8.5)
    fig.text(0.01, 0.005, f"Cost on clean data (all {c['patients']:,} held-out test patients): {c['withheld_pct']*100:.2f}% of patient-hours withheld "
             f"({c['withheld_hours']:,}/{c['hours']:,}); {c['patients_ever_withheld_pct']*100:.0f}% of patients have ≥1 withheld hour "
             f"(median {c['median_withheld_hours_per_affected_patient']:.0f} h).", fontsize=8, color=INK2)
    fig.subplots_adjust(right=0.78)
    save(fig, "10_abstention.png")


if __name__ == "__main__" and (RES / "experiments2.json").exists():
    E = json.load(open(RES / "experiments2.json"))
    baselines_fig(E); threshold_fig(E)
    if (RES / "abstention.json").exists():
        abstention_fig(json.load(open(RES / "abstention.json")))


def subgroups_ci_fig(rows, overall_auroc, overall_sens):
    d = pd.DataFrame(rows)
    lab = {"hospital": "Hospital", "age_group": "Age", "sex": "Sex", "icu": "ICU"}
    order = {"<45": 0, "45-64": 1, "65-79": 2, "80+": 3}
    d["o"] = d.apply(lambda r: (list(lab).index(r["attribute"]), order.get(r["group"], 0), r["group"]), axis=1)
    d = d.sort_values("o").reset_index(drop=True)
    d["name"] = d.apply(lambda r: f"{lab[r['attribute']]}: {r['group']}  (n={r['n_patients']:,}, septic={r['n_septic']})", axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), sharey=True)
    y = np.arange(len(d))[::-1]
    for ax, key, ci, ov, title, xl in ((axes[0], "auroc", "auroc_ci", overall_auroc, "AUROC", (0.76, 0.92)),
                                       (axes[1], "patient_sensitivity", "sens_ci", overall_sens, "Sepsis patients alerted", (0.55, 0.95))):
        lo = d[ci].map(lambda v: v[0]); hi = d[ci].map(lambda v: v[1])
        ax.axvline(ov, color=GRAY, ls="--", lw=1)
        ax.hlines(y, lo, hi, color="#9ec5f4", lw=3)
        ax.plot(d[key], y, "o", color=BLUE, ms=7, markeredgecolor="white", markeredgewidth=1.5)
        ax.set_xlim(*xl); ax.set_title(title, fontsize=11)
        if key == "patient_sensitivity":
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_yticks(y, d["name"], fontsize=8.5)
    fig.suptitle("5 · Subgroup audit with 95% bootstrap intervals (dashed = overall)", x=0.02, ha="left",
                 fontweight="bold", color=INK)
    save(fig, "5_subgroups.png")


if __name__ == "__main__" and (RES / "subgroups_ci.json").exists():
    _R = json.load(open(RES / "results.json"))["main"]["calibrated"]
    subgroups_ci_fig(json.load(open(RES / "subgroups_ci.json")), _R["auroc"], _R["patient_sensitivity"])
