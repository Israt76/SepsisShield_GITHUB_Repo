"""Drive the running dashboard and assert the trust-aware abstention behaviour end to end.

Usage: streamlit run app/app.py  (in another shell), then  python tools/dashboard_abstention_check.py
"""
import asyncio
import sys
from urllib.parse import quote
from playwright.async_api import async_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
F = quote("Thermometer reports °F")
M = quote("Vitals overwritten to look normal")
CASES = [
    ("clean, trusted", "?pid=p119917&hour=58", "SHOW"),
    ("°F fault (inside window)", f"?pid=p119917&corr={F}&start=34&len=8&hour=36", "WITHHELD"),
    ("edited chart", f"?pid=p018345&corr={M}&start=46&len=12&hour=48", "WITHHELD"),
    ("edited chart, later hour", f"?pid=p018345&corr={M}&start=46&len=12&hour=57", "WARN"),
]


async def main():
    ok = True
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1600, "height": 1000})
        for name, q, expect in CASES:
            await pg.goto(BASE + "/" + q, wait_until="networkidle")
            await pg.wait_for_selector("text=Risk trajectory", timeout=90000)
            await pg.wait_for_timeout(2500)
            t = await pg.inner_text("body")
            withheld = "Withheld" in t and "Not issued" in t
            warned = (not withheld) and "Verify inputs before acting" in t
            shown = (not withheld) and (not warned) and "Inputs look reliable" in t
            got = "WITHHELD" if withheld else "WARN" if warned else "SHOW" if shown else "?"
            passed = got == expect and "Traceback" not in t
            ok &= passed
            print(f"{'PASS' if passed else 'FAIL'}  {name:28s} expected {expect:8s} got {got}")
        await b.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
