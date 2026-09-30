import asyncio
from urllib.parse import quote
from playwright.async_api import async_playwright
from capture import shot, OUT
M = quote("Vitals overwritten to look normal"); F = quote("Thermometer reports °F")
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        B2 = "http://localhost:8501/?pid=p018345"
        await shot(pg, f"{B2}&hour=48", "m2_clean_h48")
        await shot(pg, f"{B2}&corr={M}&start=46&len=12&hour=48", "m2_mask_h48", scrolls=(0, 560))
        await shot(pg, f"{B2}&hour=57", "m2_clean_h57")
        await shot(pg, f"{B2}&corr={M}&start=46&len=12&hour=57", "m2_mask_h57")
        await shot(pg, f"http://localhost:8501/?pid=p119917&corr={F}&start=34&len=8&hour=36", "fahr36", scrolls=(0, 560))
        await b.close()
asyncio.run(main())
