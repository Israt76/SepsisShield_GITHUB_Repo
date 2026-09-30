"""Build the subtitle-band version: 1920x1080 picture untouched on top + 120 px black band with captions.

Steps (from tools/video/):
  1. SS_BAND=1 python render.py            -> SepsisShield_demo_band_main.mp4 (main cut, captions in band)
  2. python make_band_video.py             -> intro frames without captions, band composed, joined with the main cut
Output: SepsisShield_demo_subtitle_band.mp4 (1920x1200)
"""
import asyncio, subprocess
from pathlib import Path
from PIL import Image
from playwright.async_api import async_playwright
from band import with_band, W, H, BAND

HERE = Path(__file__).resolve().parent
T_GLITCH, T_RISK, T_TRUST, DUR = 2.3, 3.2, 4.8, 7.5
CAPS = [(0.4, T_GLITCH, "ICU patient · vital signs streaming to an early-warning model"),
        (T_GLITCH, T_RISK, "The thermometer starts reporting °F into a °C field"),
        (T_RISK, T_TRUST, "The risk score rises — driven by a reading that cannot be real"),
        (T_TRUST, DUR, "SepsisShield checks the inputs before trusting the prediction")]
ease = lambda x: (lambda y: y * y * (3 - 2 * y))(min(max(x, 0), 1))


async def capture(out):
    out.mkdir(exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        await pg.goto((HERE / "intro.html").as_uri()); await pg.wait_for_function("window.READY===true")
        await pg.evaluate("window.SHOW_CAPTIONS=false")
        for f in range(int(DUR * 30)):
            await pg.evaluate(f"window.render({f / 30})"); await pg.screenshot(path=str(out / f"f{f:04d}.png"))
        await b.close()


if __name__ == "__main__":
    frames = HERE / "intro_frames_nocap"
    asyncio.run(capture(frames))
    subprocess.run(["python3", "intro_audio.py"], check=True, cwd=HERE)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H + BAND}", "-r", "30",
                          "-i", "-", "-i", "intro_audio.wav", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                          "-b:a", "160k", "-ar", "48000", "-ac", "1", "-shortest", "intro_band.mp4"], stdin=subprocess.PIPE, cwd=HERE)
    for f in range(int(DUR * 30)):
        t = f / 30; cap, a = None, 0
        for s0, s1, txt in CAPS:
            u = min(ease((t - s0) / 0.25), ease((s1 - t) / 0.25))
            if u > 0: cap, a = txt, u
        p.stdin.write(with_band(Image.open(frames / f"f{f:04d}.png"), cap, a).tobytes())
    p.stdin.close(); p.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "intro_band.mp4", "-i", "SepsisShield_demo_band_main.mp4", "-filter_complex",
                    "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart",
                    "SepsisShield_demo_subtitle_band.mp4"], check=True, cwd=HERE)
    print("wrote SepsisShield_demo_subtitle_band.mp4")
