"""Render the 7.5 s clinical intro (intro.html) and prepend it to the main demo video.

Usage (from tools/video/):  python render_intro.py
Needs: playwright (chromium), ffmpeg. Inputs: intro.html, intro_audio.py, SepsisShield_demo.mp4 (main cut from render.py).
Output: SepsisShield_demo_with_intro.mp4
"""
import asyncio
import shutil
import subprocess
from pathlib import Path
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
FPS, DUR = 30, 7.5


async def capture():
    frames = HERE / "intro_frames"
    shutil.rmtree(frames, ignore_errors=True); frames.mkdir()
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        await pg.goto((HERE / "intro.html").as_uri())
        await pg.wait_for_function("window.READY===true")
        for f in range(int(FPS * DUR)):
            await pg.evaluate(f"window.render({f / FPS})")
            await pg.screenshot(path=str(frames / f"f{f:04d}.png"))
        await b.close()


def run(*cmd):
    subprocess.run(cmd, check=True, cwd=HERE)


if __name__ == "__main__":
    asyncio.run(capture())
    run("python3", "intro_audio.py")
    run("ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", "intro_frames/f%04d.png", "-i", "intro_audio.wav",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-ar", "48000", "-ac", "1", "-shortest", "intro.mp4")
    run("ffmpeg", "-v", "error", "-y", "-i", "intro.mp4", "-i", "SepsisShield_demo.mp4", "-filter_complex",
        "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset",
        "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart", "SepsisShield_demo_with_intro.mp4")
    print("wrote SepsisShield_demo_with_intro.mp4")
