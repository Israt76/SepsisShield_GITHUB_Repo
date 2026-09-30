"""Render the SepsisShield demo video: frames + narration + burned-in captions -> MP4 (1920x1080, 30 fps)."""
import json, subprocess, wave, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = Path(__file__).resolve().parent
FR = HERE / "frames"
FIG = HERE.parents[1] / "results" / "figures"
OUT = HERE / "SepsisShield_demo.mp4"
W, H, FPS = 1920, 1080, 30
GAP_LINE, GAP_SCENE, LEAD, TAIL = 0.30, 0.85, 0.8, 3.5

BG, INK, INK2, BLUE, RED, GREEN, AMBER = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#d03b3b", "#0ca30c", "#b37d00"
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(sz, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, sz, index=0)


# ------------------------------------------------------------------ timeline
meta = json.load(open(HERE / "audio" / "meta.json"))
t = LEAD
scenes, lines = {}, []
prev_scene = None
for m in meta:
    if prev_scene is not None and m["scene"] != prev_scene:
        t += GAP_SCENE - GAP_LINE
    if m["scene"] not in scenes:
        scenes[m["scene"]] = [t, None]
    m["start"], m["end"] = t, t + m["dur"]
    lines.append(m)
    t = m["end"] + GAP_LINE
    prev_scene = m["scene"]
order = list(scenes)
for i, s in enumerate(order):
    scenes[s][1] = scenes[order[i + 1]][0] if i + 1 < len(order) else t + TAIL
scenes[order[0]][0] = 0.0
TOTAL = scenes[order[-1]][1]


def line_times(scene):
    return [(m["start"], m["end"]) for m in lines if m["scene"] == scene]


# ------------------------------------------------------------------ helpers
_cache = {}


def img(name):
    if name not in _cache:
        _cache[name] = Image.open(FR / name).convert("RGB") if (FR / name).exists() else Image.open(name).convert("RGB")
    return _cache[name]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp_box(a, b, u):
    return tuple(a[i] + (b[i] - a[i]) * u for i in range(4))


def view(im, box):
    """Crop box (x0,y0,x1,y1) from image and fit to 16:9 frame."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    # enforce 16:9 around centre
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if bw / bh > W / H:
        bh = bw * H / W
    else:
        bw = bh * W / H
    bw, bh = min(bw, im.width), min(bh, im.height)
    x0 = min(max(cx - bw / 2, 0), im.width - bw)
    y0 = min(max(cy - bh / 2, 0), im.height - bh)
    x1, y1 = x0 + bw, y0 + bh
    return im.resize((W, H), Image.BICUBIC, box=(x0, y0, x1, y1))


def blend(a, b, u):
    return a if u <= 0 else b if u >= 1 else Image.blend(a, b, u)


MAIN = (480, 200, 1740, 910)          # dashboard top: tiles + risk chart
TILES_L = (480, 290, 1110, 470)       # prediction tiles (risk, alert)
TILES_R = (1110, 290, 1740, 470)      # reliability tiles (input trust, model confidence)
CHART = (480, 480, 1740, 820)
SHAP = (480, 285, 1150, 660)          # in scrolled (_s1) frames
INTEG1 = (1150, 285, 1740, 470)
INTEG2 = (1150, 285, 1740, 530)
FULL = (0, 0, 1920, 1080)


def card_bg():
    return Image.new("RGB", (W, H), BG)


def shield(d, cx, cy, s, color=BLUE):
    pts = [(cx - s, cy - s * 0.9), (cx, cy - s * 1.15), (cx + s, cy - s * 0.9), (cx + s, cy - s * 0.1),
           (cx + s * 0.55, cy + s * 0.75), (cx, cy + s * 1.1), (cx - s * 0.55, cy + s * 0.75), (cx - s, cy - s * 0.1)]
    d.polygon(pts, fill=color)
    d.line([(cx - s * 0.42, cy - s * 0.05), (cx - s * 0.1, cy + s * 0.3), (cx + s * 0.48, cy - s * 0.38)],
           fill="white", width=max(4, int(s * 0.16)), joint="curve")


def centered(d, y, text, f, fill=INK):
    w = d.textlength(text, font=f)
    d.text(((W - w) / 2, y), text, font=f, fill=fill)


def figure_card(path, title):
    c = card_bg(); d = ImageDraw.Draw(c)
    d.text((110, 70), title, font=font(46, True), fill=INK)
    f = Image.open(path).convert("RGB")
    maxw, maxh = W - 220, H - 290
    s = min(maxw / f.width, maxh / f.height)
    f = f.resize((int(f.width * s), int(f.height * s)), Image.LANCZOS)
    c.paste(f, ((W - f.width) // 2, 165 + (maxh - f.height) // 2))
    return c


# ------------------------------------------------------------------ static cards
def make_problem():
    c = card_bg(); d = ImageDraw.Draw(c)
    centered(d, 300, "AI early-warning systems", font(64, True))
    centered(d, 390, "trust whatever data they are given.", font(64, True))
    centered(d, 520, "°F thermometers · wrong lab units · frozen monitors · edited charts", font(34), INK2)
    return c


def make_title():
    c = card_bg(); d = ImageDraw.Draw(c)
    shield(d, W / 2, 330, 95)
    centered(d, 470, "SepsisShield AI", font(96, True))
    centered(d, 610, "Predict early.  Explain clearly.  Know when not to trust the model.", font(40), INK2)
    return c


def make_results():
    c = card_bg(); d = ImageDraw.Draw(c)
    d.text((110, 70), "Held-out test set · 8,068 patients never used in training or tuning", font=font(44, True), fill=INK)
    stats = [("0.852", "AUROC"), ("0.425", "Challenge utility\n(winner's CV: 0.430)"),
             ("79%", "of sepsis patients\nalerted"), ("54%", "alerted ≥ 6 h\nbefore onset")]
    tw, gap, x = 390, 36, 110
    for v, lab in stats:
        d.rounded_rectangle((x, 250, x + tw, 640), 22, fill="white", outline="#e6e5e1", width=3)
        d.text((x + 36, 290), v, font=font(110, True), fill=BLUE)
        d.multiline_text((x + 40, 460), lab, font=font(36), fill=INK2, spacing=10)
        x += tw + gap
    d.text((110, 720), "Also: calibration error 0.28 pp · external validation AUROC 0.79 / 0.78 · subgroup audit",
           font=font(34), fill=INK2)
    return c


def make_end():
    c = card_bg(); d = ImageDraw.Draw(c)
    shield(d, W / 2, 250, 80)
    centered(d, 370, "SepsisShield AI", font(88, True))
    centered(d, 500, "Predict early.  Explain clearly.  Know when not to trust the model.", font(40), INK2)
    centered(d, 640, "Built by Israt Jahan Aunika · Global Innovation Build Challenge V2", font(34), INK)
    centered(d, 700, "Data: PhysioNet/CinC Challenge 2019 (CC BY 4.0) · Research prototype — not a medical device",
             font(28), INK2)
    return c


def make_arch():
    c = card_bg()
    f = Image.open(FIG / "0_architecture.png").convert("RGB")
    sc = min((W - 120) / f.width, (H - 120) / f.height)
    f = f.resize((int(f.width * sc), int(f.height * sc)), Image.LANCZOS)
    c.paste(f, ((W - f.width) // 2, (H - f.height) // 2 - 30))
    return c


def make_end3():
    c = card_bg(); d = ImageDraw.Draw(c)
    shield(d, W / 2, 175, 62)
    centered(d, 265, "SepsisShield AI", font(78, True))
    centered(d, 375, "Predict early.  Explain clearly.  Know when not to trust the model.", font(36), INK2)
    stats = [("0.852", "AUROC", "held-out test set\n8,068 patients"),
             ("79%", "of sepsis patients", "alerted (12 h before to\n3 h after onset)"),
             ("95%", "of alert-changing accidental", "data faults flagged or withheld\n(6,481 / 6,790)")]
    tw, gap = 500, 40
    x = (W - (3 * tw + 2 * gap)) / 2
    for v, l1, l2 in stats:
        d.rounded_rectangle((x, 460, x + tw, 800), 22, fill="white", outline="#e6e5e1", width=3)
        vw = d.textlength(v, font=font(96, True)); d.text((x + (tw - vw) / 2, 480), v, font=font(96, True), fill=BLUE)
        lw = d.textlength(l1, font=font(34, True)); d.text((x + (tw - lw) / 2, 612), l1, font=font(34, True), fill=INK)
        for i, ln in enumerate(l2.split("\n")):
            w_ = d.textlength(ln, font=font(28)); d.text((x + (tw - w_) / 2, 666 + i * 40), ln, font=font(28), fill=INK2)
        x += tw + gap
    centered(d, 830, "Deliberately edited inputs: 42.5% (334 / 786)  ·  Clean predictions withheld: 0.46%", font(30), INK2)
    centered(d, 905, "Built by Israt Jahan Aunika · Global Innovation Build Challenge V2 · Track 02", font(28), INK)
    centered(d, 950, "Data: PhysioNet/CinC Challenge 2019 (CC BY 4.0) · Research prototype — not a medical device",
             font(24), INK2)
    return c


def evidence_card(k):
    """k = number of rows revealed (1..3)."""
    c = card_bg(); d = ImageDraw.Draw(c)
    d.text((110, 110), "Corruption benchmark · 3,086 held-out test patients · 6 simulated input failures", font=font(38, True), fill=INK)
    rows = [("95.4%", BLUE, "of alert-changing accidental data faults flagged or withheld",
             "6,481 / 6,790 patient-hours  ·  validation-cohort replication 94.7%"),
            ("0.46%", BLUE, "of clean predictions withheld",
             "1,438 / 309,270 patient-hours (8,068 test patients)  ·  12% of patients ever, median 1 h"),
            ("42.5%", "#b3461d", "of deliberately edited inputs flagged or withheld",
             "334 / 786 patient-hours  ·  substantially weaker — our main limitation")]
    y = 195
    for i, (v_, col, l1, l2) in enumerate(rows[:k]):
        d.rounded_rectangle((110, y, W - 110, y + 178), 22, fill="white", outline="#e6e5e1", width=3)
        d.text((150, y + 22), v_, font=font(104, True), fill=col)
        d.text((600, y + 38), l1, font=font(38, True), fill=INK)
        d.text((600, y + 102), l2, font=font(28), fill=INK2)
        y += 196
    d.text((110, 790), "Alert-changing fault = a patient-hour where the fault flipped the alert decision to a wrong one", font=font(26), fill=INK2)
    d.text((110, 826), "(suppressed a timely sepsis alert, or created an unwarranted one).", font=font(26), fill=INK2)
    return c


def make_problem2():
    c = card_bg(); d = ImageDraw.Draw(c)
    centered(d, 330, "Most early-warning models assume", font(66, True))
    centered(d, 420, "their inputs are correct.", font(66, True))
    centered(d, 560, "°F thermometers · wrong lab units · frozen monitors · edited charts", font(34), INK2)
    return c


CARDS = {"problem": make_problem2(), "title": make_title(), "arch": make_arch(), "end": make_end3(),
         "ev1": evidence_card(1), "ev2": evidence_card(2), "ev3": evidence_card(3),
         "cross": figure_card(FIG / "3_cross_hospital.png", "Where it degrades: an unseen hospital")}

CHIPS = {"monitor": "Act 1 · Prediction", "explain": "Act 1 · Prediction", "fahr": "Act 2 · Failure: °F thermometer",
         "mask": "Act 2 · Failure: edited chart", "response": "Act 3 · Trust-aware response",
         "evidence": "Act 3 · Evidence", "limits": "Limitations"}


# ------------------------------------------------------------------ scene renderers (local time u in seconds)
def _st(scene, lt):
    base = scenes[scene][0]
    return [x[0] - base for x in lt]


def sc_title(u, lt):
    s = _st("title", lt)
    fr = blend(CARDS["problem"], CARDS["title"], ease((u - (s[1] - 0.3)) / 0.8))
    return blend(fr, CARDS["arch"], ease((u - (s[2] - 0.3)) / 0.8))


def sc_monitor(u, lt):
    s = _st("monitor", lt)
    D = scenes["monitor"][1] - scenes["monitor"][0]
    hr = np.interp(u, [1.5, s[1] + 2.0, s[1] + 5.5, s[1] + 7.5], [8, 48, 56, 60])
    h = int(min(60, max(8, 2 * round(hr / 2))))
    return view(img(f"monitor_h{h:02d}.png"), MAIN)


def sc_explain(u, lt):
    s = _st("explain", lt)
    a = view(img("monitor_final_s1.png"), lerp_box(FULL, SHAP, ease((u - 0.2) / 1.2)))
    b = view(img("monitor_final_s0.png"), TILES_R)
    return blend(a, b, ease((u - (s[1] - 0.2)) / 0.35))


def sc_fahr(u, lt):
    s = _st("fahr", lt)
    fr = view(img("fahr36_s0.png"), lerp_box(MAIN, CHART, ease((u - s[1]) / 1.2)))
    tiles = view(img("fahr36_s0.png"), TILES_L)
    msg = view(img("fahr36_s1.png"), INTEG1)
    fr = blend(fr, msg, ease((u - (s[2] - 0.2)) / 0.35))
    return blend(fr, tiles, ease((u - (s[2] + 3.4)) / 0.35))


def sc_mask(u, lt):
    s = _st("mask", lt)
    clean = view(img("m2_clean_h57.png"), MAIN)
    masked = view(img("m2_mask_h57.png"), MAIN)
    h48 = view(img("m2_mask_h48_s0.png"), MAIN)
    fr = blend(clean, masked, ease((u - (s[1] + 0.8)) / 0.8))
    return blend(fr, h48, ease((u - (s[2] + 2.2)) / 0.7))


def sc_response(u, lt):
    s = _st("response", lt)
    conf = view(img("m2_mask_h48_s0.png"), lerp_box(MAIN, TILES_R, ease((u - 0.3) / 1.4)))
    msgs = view(img("m2_mask_h48_s1.png"), INTEG2)
    left = view(img("m2_mask_h48_s0.png"), TILES_L)
    fr = blend(conf, msgs, ease((u - (s[1] - 0.2)) / 0.35))
    return blend(fr, left, ease((u - (s[2] - 0.2)) / 0.35))


def sc_evidence(u, lt):
    s = _st("evidence", lt)
    fr = blend(CARDS["ev1"], CARDS["ev2"], ease((u - (s[1] - 0.1)) / 0.35))
    return blend(fr, CARDS["ev3"], ease((u - (s[2] - 0.1)) / 0.35))


RENDER = {"title": sc_title, "monitor": sc_monitor, "explain": sc_explain, "fahr": sc_fahr, "mask": sc_mask,
          "response": sc_response, "evidence": sc_evidence, "limits": lambda u, lt: CARDS["cross"],
          "end": lambda u, lt: CARDS["end"]}


# ------------------------------------------------------------------ overlays
CAP_F = font(38)
CHIP_F = font(28, True)


def wrap(d, text, f, maxw):
    words, out, cur = text.split(), [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if d.textlength(trial, font=f) <= maxw:
            cur = trial
        else:
            out.append(cur); cur = w_
    out.append(cur)
    return out


import os
BAND_MODE = os.environ.get("SS_BAND") == "1"   # captions go into a black band below the picture


def current_caption(scene, gt):
    cur = [m for m in lines if m["start"] - 0.05 <= gt <= m["end"] + GAP_LINE]
    return cur[0]["caption"] if cur and scene not in ("end",) else None


def overlay(frame, scene, gt):
    fr = frame.convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    if scene in CHIPS:
        txt = CHIPS[scene]
        tw = d.textlength(txt, font=CHIP_F)
        d.rounded_rectangle((W - 40 - tw - 44, 36, W - 40, 36 + 58), 29, fill=(42, 120, 214, 235))
        d.text((W - 40 - tw - 22, 44), txt, font=CHIP_F, fill="white")
    cur = [m for m in lines if m["start"] - 0.05 <= gt <= m["end"] + GAP_LINE]
    if cur and scene not in ("end",) and not BAND_MODE:
        cap = cur[0]["caption"]
        ls = wrap(d, cap, CAP_F, 1500)
        lh = 54
        bh = lh * len(ls) + 34
        y0 = H - 60 - bh
        widest = max(d.textlength(l, font=CAP_F) for l in ls)
        x0 = (W - widest) / 2 - 30
        d.rounded_rectangle((x0, y0, x0 + widest + 60, y0 + bh), 18, fill=(18, 18, 18, 228))
        for i, l in enumerate(ls):
            lw = d.textlength(l, font=CAP_F)
            d.text(((W - lw) / 2, y0 + 14 + i * lh), l, font=CAP_F, fill="white")
    out = Image.alpha_composite(fr, ov).convert("RGB")
    if BAND_MODE:
        from band import with_band
        out = with_band(out, current_caption(scene, gt))
    return out


# ------------------------------------------------------------------ audio
def build_audio(path):
    with wave.open(lines[0]["file"]) as w:
        sr = w.getframerate()
    buf = np.zeros(int((TOTAL + 0.5) * sr), dtype=np.int16)
    for m in lines:
        with wave.open(m["file"]) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        i = int(m["start"] * sr)
        buf[i:i + len(a)] = a[: len(buf) - i]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(buf.tobytes())


def main(preview=None):
    apath = HERE / "audio" / "narration_full.wav"
    build_audio(apath)
    if preview:
        for name, gt in preview:
            sc = next(s for s in order if scenes[s][0] <= gt < scenes[s][1])
            fr = RENDER[sc](gt - scenes[sc][0], line_times(sc))
            overlay(fr, sc, gt).save(HERE / f"preview_{name}.png")
        return
    n = int(TOTAL * FPS)
    from band import BAND
    out_path = HERE / "SepsisShield_demo_band_main.mp4" if BAND_MODE else OUT
    size = f"{W}x{H + BAND}" if BAND_MODE else f"{W}x{H}"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", str(FPS),
           "-i", "-", "-i", str(apath), "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000", "-ar", "48000", "-c:a", "aac", "-b:a", "160k",
           "-movflags", "+faststart", "-shortest", str(out_path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(n):
        gt = f / FPS
        sc = next(s for s in order if scenes[s][0] <= gt < scenes[s][1]) if gt < TOTAL else order[-1]
        fr = RENDER[sc](gt - scenes[sc][0], line_times(sc))
        p.stdin.write(overlay(fr, sc, gt).tobytes())
        if f % 600 == 0:
            print(f"{f}/{n}", flush=True)
    p.stdin.close(); p.wait()
    print("wrote", out_path, f"{TOTAL:.1f}s")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        print({s: (round(a, 1), round(b, 1)) for s, (a, b) in scenes.items()}, "total", round(TOTAL, 1))
        pts = [("title_a", 3), ("title_b", scenes["title"][1] - 1.5)]
        for s in order[1:]:
            a, b = scenes[s]
            pts += [(f"{s}_1", a + (b - a) * 0.3), (f"{s}_2", a + (b - a) * 0.8)]
        main(preview=pts)
    else:
        main()
