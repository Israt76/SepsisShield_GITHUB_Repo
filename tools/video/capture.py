"""Capture 1920x1080 dashboard frames for the demo video (app must be running on :8501)."""
import asyncio
from pathlib import Path
from urllib.parse import quote
from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parent / "frames"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8501/?pid=p119917"
HIDE = """header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
          [data-testid="stStatusWidget"] {display:none !important}"""


async def shot(pg, url, name, scrolls=(0,)):
    await pg.goto(url, wait_until="networkidle")
    await pg.add_style_tag(content=HIDE)
    await pg.wait_for_selector("text=Risk trajectory", timeout=90000)
    await pg.wait_for_timeout(2500)
    for i, y in enumerate(scrolls):
        await pg.evaluate(f"""() => {{ const m = document.querySelector('[data-testid="stMain"]') ||
                                          document.querySelector('section.main');
                                      if (m) m.scrollTo(0, {y}); window.scrollTo(0, {y}); }}""")
        await pg.wait_for_timeout(900)
        await pg.screenshot(path=str(OUT / (f"{name}.png" if len(scrolls) == 1 else f"{name}_s{i}.png")))


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        # stop-motion: hours 8..60
        for h in range(8, 61, 2):
            await shot(pg, f"{BASE}&hour={h}", f"monitor_h{h:02d}")
            print("hour", h, flush=True)
        await shot(pg, f"{BASE}&hour=60", "monitor_final", scrolls=(0, 560))
        await shot(pg, f"{BASE}&hour=63", "clean_h63")
        await shot(pg, f"{BASE}&corr={quote('Thermometer reports °F')}&start=34&len=8&hour=44", "fahr", scrolls=(0, 560))
        await shot(pg, f"{BASE}&corr={quote('Vitals overwritten to look normal')}&start=52&len=12&hour=63", "mask",
                   scrolls=(0, 560))
        await pg.goto(BASE, wait_until="networkidle")
        await pg.add_style_tag(content=HIDE)
        await pg.wait_for_selector("text=Risk trajectory", timeout=90000)
        await pg.get_by_role("tab", name="Validation evidence").click()
        await pg.wait_for_timeout(4000)
        await pg.screenshot(path=str(OUT / "validation.png"))
        await b.close()


if __name__ == "__main__":
    asyncio.run(main())
