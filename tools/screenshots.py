"""Capture dashboard screenshots for the README / Devpost (requires the app running on :8501)."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parents[1] / "results" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


async def pick(pg, label_text, option):
    box = pg.locator(f'div[data-testid="stSelectbox"]:has-text("{label_text}")').first
    await box.click()
    await pg.wait_for_timeout(400)
    await pg.get_by_role("option").filter(has_text=option).first.click()
    await pg.wait_for_timeout(4000)


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1500, "height": 1100})
        await pg.goto("http://localhost:8501", wait_until="networkidle")
        await pg.wait_for_selector("text=Risk trajectory", timeout=90000)
        await pg.wait_for_timeout(3000)
        await pg.screenshot(path=str(OUT / "01_monitor.png"))
        await pick(pg, "Corruption", "Vitals overwritten")
        await pg.screenshot(path=str(OUT / "02_stress_masking.png"))
        await pick(pg, "Corruption", "Thermometer")
        await pg.screenshot(path=str(OUT / "03_stress_fahrenheit.png"), full_page=True)
        await pg.get_by_role("tab", name="Validation evidence").click()
        await pg.wait_for_timeout(4000)
        await pg.screenshot(path=str(OUT / "04_validation.png"), full_page=True)
        print("error on page:", "Traceback" in await pg.inner_text("body"))
        await b.close()


if __name__ == "__main__":
    asyncio.run(main())
