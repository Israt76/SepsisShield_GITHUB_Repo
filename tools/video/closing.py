"""Render the 2 s closing frame and append it to the video.  Usage: python closing.py <input.mp4> <output.mp4>"""
import subprocess, sys
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
f = lambda s, b=False: ImageFont.truetype(B if b else R, s, index=0)
c = Image.new("RGB", (W, H), "#fcfcfb"); d = ImageDraw.Draw(c)


def centered(y, t, font, fill):
    d.text(((W - d.textlength(t, font=font)) / 2, y), t, font=font, fill=fill)


cx, cy, s = W / 2, 300, 70
d.polygon([(cx - s, cy - s * .9), (cx, cy - s * 1.15), (cx + s, cy - s * .9), (cx + s, cy - s * .1), (cx + s * .55, cy + s * .75),
           (cx, cy + s * 1.1), (cx - s * .55, cy + s * .75), (cx - s, cy - s * .1)], fill="#2a78d6")
d.line([(cx - s * .42, cy - s * .05), (cx - s * .1, cy + s * .3), (cx + s * .48, cy - s * .38)], fill="white", width=11, joint="curve")
centered(410, "SepsisShield AI", f(96, True), "#0b0b0b")
centered(555, "Predict early. Explain clearly. Know when not to trust the model.", f(42), "#52514e")
parts = [("0.852", "#2a78d6", True), (" AUROC", "#0b0b0b", False), ("   •   ", "#8a8984", False), ("79%", "#2a78d6", True),
         (" patients alerted", "#0b0b0b", False), ("   •   ", "#8a8984", False), ("95.4%", "#2a78d6", True),
         (" accidental dangerous faults flagged/withheld", "#0b0b0b", False)]
fb, fr = f(42, True), f(36)
x = (W - sum(d.textlength(t, font=fb if b else fr) for t, _, b in parts)) / 2
for t, col, b in parts:
    fnt = fb if b else fr; d.text((x, 700 + (0 if b else 5)), t, font=fnt, fill=col); x += d.textlength(t, font=fnt)
centered(785, "95.4%: alert-changing accidental data faults in our corruption benchmark (6,481 / 6,790)  ·  "
              "deliberately edited inputs: 42.5% (334 / 786)", f(25), "#8a8984")
centered(960, "Research prototype on de-identified PhysioNet 2019 data · not a medical device", f(24), "#8a8984")
c.save("closing_frame.png")

src, dst = sys.argv[1], sys.argv[2]
run = lambda *a: subprocess.run(a, check=True)
run("ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "30", "-t", "2", "-i", "closing_frame.png", "-f", "lavfi", "-t", "2",
    "-i", "anullsrc=r=48000:cl=mono", "-vf", "fade=in:st=0:d=0.3:color=0xfcfcfb,format=yuv420p", "-c:v", "libx264", "-crf", "18",
    "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "1", "-shortest", "closing.mp4")
run("ffmpeg", "-v", "error", "-y", "-i", src, "-i", "closing.mp4", "-filter_complex", "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
    "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "160k",
    "-ar", "48000", "-movflags", "+faststart", dst)
