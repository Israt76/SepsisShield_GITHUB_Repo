"""SepsisShield AI — Streamlit dashboard.

Run:  streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
import pipeline  # noqa: E402
import integrity  # noqa: E402
from labels import pretty  # noqa: E402

# palette (validated reference palette from the dataviz method)
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#8a8984"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
TRUST_STYLE = {"HIGH": (GOOD, "✅", "Inputs look reliable"),
               "REDUCED": (WARN, "⚠️", "Verify inputs before acting"),
               "LOW": (CRIT, "⛔", "Do not trust this prediction")}

st.set_page_config(page_title="SepsisShield AI", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.tile {border: 1px solid #e6e5e1; border-radius: 10px; padding: 14px 16px; background: #fcfcfb; height: 100%;}
.tile .lbl {font-size: 0.78rem; color: #52514e; text-transform: uppercase; letter-spacing: .04em;}
.tile .val {font-size: 1.9rem; font-weight: 650; color: #0b0b0b; line-height: 1.2; margin-top: 2px;}
.tile .sub {font-size: 0.82rem; color: #52514e; margin-top: 2px;}
.badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-weight: 600; font-size: .95rem;}
.msg {border-left: 4px solid; padding: 8px 12px; margin: 6px 0; background: #fcfcfb; border-radius: 0 6px 6px 0; font-size: .9rem; color:#0b0b0b}
.disclaimer {font-size: .78rem; color: #52514e;}
.grp {font-size: .72rem; font-weight: 700; letter-spacing: .08em; color: #52514e; margin-bottom: -6px;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def demo_patients():
    return pd.read_parquet(ROOT / "app" / "demo_patients.parquet")


@st.cache_data
def demo_index():
    return pd.read_parquet(ROOT / "app" / "demo_index.parquet")


@st.cache_data(show_spinner=False)
def run(raw: pd.DataFrame):
    return pipeline.score(raw)


def apply_corruption(raw, kind, start, length):
    raw = raw.copy()
    m = (raw["hour"] >= start) & (raw["hour"] < start + length)
    if kind == "Thermometer reports °F":
        raw.loc[m, "Temp"] = raw.loc[m, "Temp"] * 1.8 + 32
    elif kind == "Lab unit mix-up (SI units)":
        raw.loc[m, "Creatinine"] *= 88.4
        raw.loc[m, "Glucose"] /= 18.0
        raw.loc[m, "Lactate"] *= 9.0
    elif kind == "Monitor artifact":
        rng = np.random.default_rng(0)
        hit = m & (rng.random(len(raw)) < 0.5)
        raw.loc[hit, "HR"] = rng.uniform(190, 240, hit.sum()).round()
        raw.loc[hit, "SBP"] = rng.uniform(25, 45, hit.sum()).round()
    elif kind == "Frozen monitor feed":
        for c in ["HR", "SBP", "MAP", "DBP", "Resp", "O2Sat", "Temp"]:
            base = raw.loc[raw["hour"] <= start, c].ffill()
            v = base.iloc[-1] if len(base) else np.nan
            raw.loc[m, c] = v
    elif kind == "Vitals overwritten to look normal":
        raw.loc[m, "HR"] -= 25; raw.loc[m, "Resp"] -= 8; raw.loc[m, "Temp"] -= 1.2
        raw.loc[m, "SBP"] += 20; raw.loc[m, "MAP"] += 15
        raw.loc[m, "Lactate"] = raw.loc[m, "Lactate"].clip(upper=1.5)
        raw.loc[m, "WBC"] = raw.loc[m, "WBC"].clip(5, 11)
    return raw


def tile(label, value, sub="", color=None):
    style = f"color:{color}" if color else ""
    return f"<div class='tile'><div class='lbl'>{label}</div><div class='val' style='{style}'>{value}</div><div class='sub'>{sub}</div></div>"


def base_layout(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=12),
                      hoverlabel=dict(bgcolor="white", font_color=INK), showlegend=False)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# ---------------------------------------------------------------- header
st.markdown("## 🛡️ SepsisShield AI")
st.markdown("**Predict early. Explain clearly. Know when not to trust the model.**  "
            "<span class='disclaimer'>Research prototype on de-identified PhysioNet 2019 data — not a medical device.</span>",
            unsafe_allow_html=True)

tab_mon, tab_val, tab_how = st.tabs(["Patient monitor", "Validation evidence", "How it works"])

# ---------------------------------------------------------------- sidebar
idx = demo_index()
qp = st.query_params  # shareable links: ?pid=p119917&corr=Monitor%20artifact&start=40&len=10&hour=60
with st.sidebar:
    st.markdown("### Patient")
    src = st.radio("Source", ["Held-out test patients", "Upload a PhysioNet .psv file"], label_visibility="collapsed")
    raw = None
    if src == "Held-out test patients":
        cats = list(idx["category"].unique())
        q_pid = qp.get("pid")
        q_cat = idx.loc[idx["patient_id"] == q_pid, "category"].iloc[0] if q_pid in set(idx["patient_id"]) else cats[0]
        cat = st.selectbox("Show", cats, index=cats.index(q_cat))
        sub = idx[idx["category"] == cat]
        ids = list(sub["patient_id"])
        pid = st.selectbox("Patient", ids, index=ids.index(q_pid) if q_pid in ids else 0,
                           format_func=lambda p: sub.set_index("patient_id").loc[p, "label"])
        raw = demo_patients()
        raw = raw[raw["patient_id"] == pid].reset_index(drop=True)
    else:
        up = st.file_uploader("PSV file (pipe-separated, PhysioNet 2019 format)", type=["psv", "txt", "csv"])
        if up is not None:
            raw = pd.read_csv(up, sep="|")
            raw.insert(0, "patient_id", Path(up.name).stem)
            raw.insert(1, "hour", np.arange(len(raw)))
    if raw is not None and len(raw):
        st.markdown("### Stress test")
        st.caption("Inject a realistic input failure and watch the integrity layer respond.")
        kinds = ["None", "Thermometer reports °F", "Lab unit mix-up (SI units)", "Monitor artifact",
                 "Frozen monitor feed", "Vitals overwritten to look normal"]
        q_corr = qp.get("corr", "None")
        kind = st.selectbox("Corruption", kinds, index=kinds.index(q_corr) if q_corr in kinds else 0)
        n = len(raw)
        if kind != "None":
            d_start = min(max(0, n - 20), max(0, n - 2))
            c_start = st.slider("Starts at hour", 0, max(0, n - 2), min(int(qp.get("start", d_start)), max(0, n - 2)))
            c_len = st.slider("Duration (hours)", 2, 24, int(qp.get("len", 10)))
            raw = apply_corruption(raw, kind, c_start, c_len)
        else:
            c_start = c_len = None

# ---------------------------------------------------------------- monitor tab
with tab_mon:
    if raw is None or not len(raw):
        st.info("Choose a patient or upload a file in the sidebar.")
    else:
        with st.spinner("Scoring..."):
            scored, feats = run(raw)
        n = len(scored)
        hour = st.slider("Hour in ICU (the model only sees data up to this hour)", 0, n - 1,
                         min(int(qp.get("hour", n - 1)), n - 1))
        row = scored.iloc[hour]
        cfg = pipeline.load()[0]
        thr = cfg["threshold"]
        onset = None
        if "SepsisLabel" in scored and scored["SepsisLabel"].max() == 1:
            onset = int(scored.loc[scored["SepsisLabel"] == 1, "hour"].min() + 6)

        tcol, ticon, ttext = TRUST_STYLE[row["trust"]]
        risk_pct = row["risk"] * 100
        withheld = row["decision"] == "WITHHELD"
        hl, hr = st.columns(2)
        hl.markdown("<div class='grp'>PREDICTION</div>", unsafe_allow_html=True)
        hr.markdown("<div class='grp'>RELIABILITY · two independent checks</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        if withheld:
            c1.markdown(tile("Sepsis risk (next ~6-12h)", "⛔ Withheld",
                             f"Inputs failed integrity checks — verify data. Research score {risk_pct:.1f}% (not for use)",
                             CRIT), unsafe_allow_html=True)
            c2.markdown(tile("Alert", "Not issued", "Prediction withheld until inputs are verified", INK2),
                        unsafe_allow_html=True)
        else:
            sub = f"Calibrated probability · alert threshold {thr*100:.1f}%"
            if row["decision"] == "SHOW_WITH_WARNING":
                sub = "⚠️ Verify inputs before acting · " + sub
            c1.markdown(tile("Sepsis risk (next ~6-12h)", f"{risk_pct:.1f}%", sub), unsafe_allow_html=True)
            c2.markdown(tile("Alert", "🔔 ALERT" if row["alert"] else "No alert",
                             "Risk above threshold" if row["alert"] else "Below threshold",
                             CRIT if row["alert"] else INK), unsafe_allow_html=True)
        c3.markdown(tile("Input trust · are the inputs believable?", f"{ticon} {row['trust']}", ttext,
                         tcol if row["trust"] != "REDUCED" else "#b37d00"), unsafe_allow_html=True)
        std_thr = pipeline.load()[3]["ens_std_thr"]
        agree = "Normal" if row["ens_std"] <= std_thr else "Unusual"
        c4.markdown(tile("Model confidence · do 5 models agree?", "Confident" if agree == "Normal" else "Uncertain",
                         f"Spread ±{row['ens_std_prob']*100:.1f} pp ({row['ens_std']:.2f} log-odds; flag > {std_thr:.2f})",
                         CRIT if agree == "Unusual" else None), unsafe_allow_html=True)
        if agree == "Normal" and row["trust"] != "HIGH":
            st.caption("The models agree with each other — but the inputs themselves look wrong. Model confidence "
                       "cannot detect bad data; that is what the integrity layer is for.")
        st.write("")

        # risk trajectory
        seen = scored.iloc[: hour + 1]
        fig = go.Figure()
        for lvl, col in (("REDUCED", WARN), ("LOW", CRIT)):
            mask = (seen["trust"] == lvl).to_numpy()
            for h in seen["hour"][mask]:
                fig.add_vrect(x0=h - 0.5, x1=h + 0.5, fillcolor=col, opacity=0.18, line_width=0, layer="below")
        if c_start is not None:
            fig.add_vrect(x0=c_start - 0.5, x1=c_start + c_len - 0.5, line=dict(color=GRAY, dash="dot", width=1),
                          fillcolor="rgba(0,0,0,0)", annotation_text="injected corruption",
                          annotation_position="bottom left", annotation_font_color=INK2)
        fig.add_hline(y=thr * 100, line=dict(color=GRAY, dash="dash", width=1),
                      annotation_text="alert threshold", annotation_position="top right",
                      annotation_font_color=INK2)
        if onset is not None:
            fig.add_vline(x=onset, line=dict(color=INK2, dash="dot", width=1.5),
                          annotation_text="sepsis onset (retrospective)", annotation_position="top left",
                          annotation_font_color=INK2)
        fig.add_trace(go.Scatter(x=seen["hour"], y=seen["risk"] * 100, mode="lines", line=dict(color=BLUE, width=2),
                                 customdata=np.stack([seen["trust"], seen["ens_std_prob"] * 100], axis=1),
                                 hovertemplate="Hour %{x}<br>Risk %{y:.1f}%<br>Trust %{customdata[0]}"
                                               "<br>Model spread ±%{customdata[1]:.2f} pp<extra></extra>"))
        fig.add_trace(go.Scatter(x=[hour], y=[risk_pct], mode="markers",
                                 marker=dict(size=10, color=BLUE, line=dict(color="white", width=2)), hoverinfo="skip"))
        fig.update_xaxes(title="Hours since ICU admission", range=[-0.5, max(n - 0.5, 10)])
        ymax = max(scored["risk"].max() * 100 * 1.15, thr * 100 * 2, 5)
        fig.update_yaxes(title="Sepsis risk (%)", range=[0, ymax])
        fig = base_layout(fig, 330)
        fig.update_layout(title=dict(text="Risk trajectory · amber = verify inputs · red = prediction withheld (low input trust)",
                                     font=dict(size=13, color=INK2), x=0))
        st.plotly_chart(fig, width="stretch")

        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("##### Why this risk? Top drivers at this hour")
            ex = pipeline.explain(feats.iloc[[hour]])
            ex = ex.iloc[::-1]
            names = [f"{pretty(f)} = {v:.4g}" if pd.notna(v) else f"{pretty(f)} = not measured"
                     for f, v in zip(ex["feature"], ex["value"])]
            fx = go.Figure(go.Bar(x=ex["shap"], y=names, orientation="h",
                                  marker=dict(color=[RED if s > 0 else BLUE for s in ex["shap"]], cornerradius=4),
                                  hovertemplate="%{y}<br>contribution %{x:+.3f} log-odds<extra></extra>"))
            fx.update_xaxes(title="← lowers risk      contribution (log-odds)      raises risk →")
            st.plotly_chart(base_layout(fx, 330), width="stretch")
            st.caption("SHAP values averaged across the 5-model ensemble. Red raises risk, blue lowers it.")
        with right:
            st.markdown("##### Input integrity at this hour")
            msgs = integrity.explain_flags(row)
            if row["ens_std"] > pipeline.load()[3]["ens_std_thr"]:
                msgs.append("The 5 models disagree more than on 99% of validation hours — unfamiliar pattern")
            if not msgs:
                st.markdown(f"<div class='msg' style='border-color:{GOOD}'>✅ All checks passed: plausible values, "
                            "consistent readings, no abrupt or coordinated shifts, live feed.</div>", unsafe_allow_html=True)
            for mtxt in msgs:
                st.markdown(f"<div class='msg' style='border-color:{tcol}'>{ticon} {mtxt}</div>", unsafe_allow_html=True)
            st.markdown("##### Over this stay so far")
            counts = {
                "Impossible values": int(seen["flag_implausible"].sum()),
                "Inconsistent readings": int(seen["flag_inconsistent"].sum()),
                "Abrupt jumps": int(seen["flag_jump"].sum()),
                "Coordinated 'normalising' shifts": int(seen["flag_coordinated_shift"].sum()),
                "Frozen-feed hours": int(seen["flag_flatline"].sum()),
            }
            st.dataframe(pd.DataFrame({"Check": counts.keys(), "Hours flagged": counts.values()}),
                         hide_index=True, width="stretch")
            tl = seen["trust"].value_counts()
            st.caption(f"Trust over {len(seen)} hours: HIGH {tl.get('HIGH', 0)} · REDUCED {tl.get('REDUCED', 0)} · LOW {tl.get('LOW', 0)}")

        st.markdown("##### Vital signs and key labs (as received)")
        panels = [("HR", "Heart rate (bpm)"), ("Temp", "Temperature (°C)"), ("MAP", "Mean arterial pressure"),
                  ("Resp", "Respiratory rate"), ("O2Sat", "O₂ saturation (%)"), ("Lactate", "Lactate (mmol/L)")]
        vf = make_subplots(rows=2, cols=3, subplot_titles=[p[1] for p in panels], horizontal_spacing=0.06,
                           vertical_spacing=0.18)
        rs = raw.iloc[: hour + 1]
        for i, (c, _) in enumerate(panels):
            r_, c_ = i // 3 + 1, i % 3 + 1
            d = rs[["hour", c]].dropna()
            vf.add_trace(go.Scatter(x=d["hour"], y=d[c], mode="lines+markers" if c == "Lactate" else "lines",
                                    line=dict(color=BLUE, width=2), marker=dict(size=7),
                                    hovertemplate=f"Hour %{{x}}<br>{c} %{{y}}<extra></extra>"), row=r_, col=c_)
            if onset is not None:
                vf.add_vline(x=onset, line=dict(color=INK2, dash="dot", width=1), row=r_, col=c_)
        vf.update_annotations(font=dict(size=12, color=INK2))
        st.plotly_chart(base_layout(vf, 420), width="stretch")

# ---------------------------------------------------------------- validation tab
with tab_val:
    res_p, ib_p = ROOT / "results" / "results.json", ROOT / "results" / "integrity_benchmark.json"
    if not res_p.exists():
        st.info("Run the training pipeline to populate validation results.")
    else:
        R = json.load(open(res_p))
        IB = json.load(open(ib_p)) if ib_p.exists() else None
        m = R["main"]["calibrated"]
        st.markdown("#### Held-out test set · "
                    f"{R['main']['n_patients']['test']:,} patients never seen in training or tuning")
        E = json.load(open(ROOT / "results" / "experiments2.json")) if (ROOT / "results" / "experiments2.json").exists() else {}
        A = json.load(open(ROOT / "results" / "abstention.json")) if (ROOT / "results" / "abstention.json").exists() else {}
        ci = E.get("bootstrap_ci", {}).get("sepsisshield", {})

        def fmt_ci(k, pct=False):
            if k not in ci:
                return ""
            lo, hi = ci[k]
            return f"95% CI {lo*100:.1f}–{hi*100:.1f}%" if pct else f"95% CI {lo:.3f}–{hi:.3f}"
        st.markdown("<div class='grp'>PREDICTION PATH</div>", unsafe_allow_html=True)
        cs = st.columns(4)
        cs[0].markdown(tile("AUROC", f"{m['auroc']:.3f}", fmt_ci("auroc")), unsafe_allow_html=True)
        cs[1].markdown(tile("Challenge utility", f"{m['utility']:.3f}", fmt_ci("utility")), unsafe_allow_html=True)
        cs[2].markdown(tile("Sepsis patients alerted", f"{m['patient_sensitivity']*100:.0f}%",
                            fmt_ci("patient_sensitivity", True)), unsafe_allow_html=True)
        cs[3].markdown(tile("External AUROC", f"{R['cross_A2B']['auroc']:.3f} / {R['cross_B2A']['auroc']:.3f}",
                            "train A → test B / train B → test A"), unsafe_allow_html=True)
        if A:
            acc, mk, cc = A["dfc_accidental"]["all"], A["dfc_masking"]["all"], A["clean_cost"]
            st.markdown("<div class='grp' style='margin-top:14px'>TRUST PATH</div>", unsafe_allow_html=True)
            ts = st.columns(4)
            ts[0].markdown(tile("Alert-changing accidental faults flagged or withheld", f"{acc['coverage']*100:.1f}%",
                                f"{acc['withheld']+acc['flagged']:,} / {acc['n']:,} patient-hours in the corruption benchmark"),
                           unsafe_allow_html=True)
            ts[1].markdown(tile("Withheld outright", f"{acc['withheld_share']*100:.0f}%",
                                f"{acc['withheld']:,} of {acc['n']:,} (LOW trust → prediction withheld)"), unsafe_allow_html=True)
            ts[2].markdown(tile("Deliberately edited inputs", f"{mk['coverage']*100:.1f}%",
                                f"{mk['withheld']+mk['flagged']} / {mk['n']} · substantially weaker — principal limitation"),
                           unsafe_allow_html=True)
            cf = A.get("clean_cost_full_test", cc)
            ts[3].markdown(tile("Clean predictions withheld", f"{cf['withheld_pct']*100:.2f}%",
                                f"of patient-hours, all {cf.get('patients', 0):,} test patients · "
                                f"{cf.get('patients_ever_withheld_pct', 0)*100:.0f}% of patients ever (median 1 h)"),
                           unsafe_allow_html=True)
        st.write("")
        figdir = ROOT / "results" / "figures"
        figs = sorted(figdir.glob("*.png"), key=lambda p: int(p.name.split("_")[0]))
        st.image(str(figs[0]), width="stretch")
        for i in range(1, len(figs), 2):
            cols = st.columns(2)
            for c, f in zip(cols, figs[i:i + 2]):
                c.image(str(f), width="stretch")
        if E:
            st.markdown("#### Baselines (same test set)")
            st.dataframe(pd.DataFrame(E["baselines"]).drop(columns=["threshold"], errors="ignore").round(3),
                         hide_index=True, width="stretch")
            st.markdown("#### Local recalibration at the unseen hospital (retrospective)")
            st.dataframe(pd.DataFrame([{"Transfer": r["transfer"], "ECE as-is (pp)": r["as_is"]["ece"] * 100,
                                        "ECE recalibrated (pp)": r["recalibrated"]["ece"] * 100,
                                        "Utility as-is": r["as_is"]["utility"], "Utility recalibrated": r["recalibrated"]["utility"]}
                                       for r in E["recalibration"]]).round(3), hide_index=True, width="stretch")
        if A:
            st.markdown("#### Trust-aware abstention by scenario (patient-hours where the fault flipped the alert decision to a wrong one)")
            st.dataframe(pd.DataFrame([{"Scenario": s["corruption"], "Dangerous failures": s["all"]["n"],
                                        "Withheld": s["all"]["withheld"], "Flagged": s["all"]["flagged"],
                                        "Silent": s["all"]["silent"]} for s in A["scenarios"]]),
                         hide_index=True, width="stretch")
        st.markdown("#### Cross-hospital generalization")
        rows = []
        for k, name in (("cross_A2B", "Train Hospital A → test Hospital B"), ("cross_B2A", "Train Hospital B → test Hospital A")):
            if k in R:
                x = R[k]
                rows.append({"Setting": name, "Internal AUROC": x["internal_val"]["auroc"], "External AUROC": x["auroc"],
                             "Internal utility": x["internal_val"]["utility"], "External utility": x["utility"]})
        if rows:
            st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        if "ablation" in R:
            st.markdown("#### Ablation: what each feature group adds")
            st.dataframe(pd.DataFrame(R["ablation"]).round(3), hide_index=True, width="stretch")
        if "subgroups" in R:
            st.markdown("#### Subgroup performance")
            st.dataframe(pd.DataFrame(R["subgroups"]).round(3), hide_index=True, width="stretch")
        if IB:
            st.markdown("#### Integrity-layer corruption benchmark")
            st.dataframe(pd.DataFrame(IB["corruptions"]).round(3), hide_index=True, width="stretch")

# ---------------------------------------------------------------- how tab
with tab_how:
    st.markdown("""
#### Pipeline
1. **Raw hourly data** — 8 vital signs, 26 labs, 6 demographic/context fields (PhysioNet 2019 format).
2. **Input-integrity layer** (runs on raw inputs, independent of the model): physiological plausibility,
   internal consistency, abrupt jumps, coordinated "normalising" shifts, frozen feeds.
3. **Causal feature engineering** — 172 features using only data up to the current hour: last values,
   informative missingness, 6h/12h trends, lab trajectories, SIRS/qSOFA/SOFA-style composites.
4. **5-model LightGBM ensemble** → **isotonic calibration** → risk %, with ensemble spread as model uncertainty.
5. **Alert threshold** chosen on the validation set to maximise the official challenge utility.
6. **Trust level** = input checks + model disagreement → HIGH / REDUCED / LOW.
7. **SHAP explanations** for every hour.

#### What the trust levels mean
- ✅ **HIGH** — inputs passed every check; the risk score can be read as calibrated.
- ⚠️ **REDUCED** — a recent input problem, a frozen feed, or unusual model disagreement. Verify inputs.
- ⛔ **LOW** — impossible or contradictory values, or a pattern consistent with manipulation. The prediction is
  **withheld** ("requires data verification"); the research score is kept for audit but not presented as actionable.

#### Limits
Retrospective data from two US hospitals; sepsis labels follow the challenge's Sepsis-3 based definition.
Not prospectively validated, not a medical device, and not a substitute for clinical judgement.
""")
