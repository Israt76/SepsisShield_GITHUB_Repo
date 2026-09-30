"""Subtitle band: extend a 1920x1080 frame downward with a black band and set the caption inside it."""
from PIL import Image, ImageDraw, ImageFont

BAND = 120
W, H = 1920, 1080
_F = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_fonts = {}


def _font(sz):
    if sz not in _fonts:
        _fonts[sz] = ImageFont.truetype(_F, sz, index=0)
    return _fonts[sz]


def _wrap(d, text, f, maxw):
    words, out, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=f) <= maxw:
            cur = trial
        else:
            out.append(cur); cur = w
    out.append(cur)
    return out


def with_band(frame: Image.Image, caption: str | None, alpha: float = 1.0) -> Image.Image:
    """Return a W x (H + BAND) image: original frame untouched on top, caption centred in the black band."""
    canvas = Image.new("RGB", (W, H + BAND), (0, 0, 0))
    canvas.paste(frame.convert("RGB"), (0, 0))
    if caption and alpha > 0:
        d = ImageDraw.Draw(canvas)
        for sz, lh in ((33, 43), (30, 39), (28, 36)):          # shrink only if a caption needs >2 lines
            f = _font(sz)
            lines = _wrap(d, caption, f, W - 240)
            if len(lines) <= 2:
                break
        total = lh * len(lines)
        y0 = H + (BAND - total) / 2 - 4
        col = tuple(int(255 * alpha) for _ in range(3))
        for i, ln in enumerate(lines):
            lw = d.textlength(ln, font=f)
            d.text(((W - lw) / 2, y0 + i * lh), ln, font=f, fill=col)
    return canvas
