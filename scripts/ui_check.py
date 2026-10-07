"""Headless-browser verification of the running dashboard (needs `pip install playwright` + a Chromium).

Usage: uvicorn app.main:app --port 8000 &  python scripts/ui_check.py [--url http://127.0.0.1:8000] [--shots DIR]
Exits non-zero if any check fails. Checks real behaviour: JS-driven prediction, comparison, batch upload+download,
error-table filtering, theme persistence, mobile layout (no horizontal overflow, collapsible nav)."""
from __future__ import annotations

import argparse
import glob
import io
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--shots", default=None)
    a = ap.parse_args()
    exe = next(iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")), None)
    shots = Path(a.shots) if a.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        ctx = b.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        # the intentionally malformed CSV upload below produces one expected 400 console line
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "status of 400" not in m.text else None)
        failed_assets: list[str] = []
        page.on("response", lambda r: failed_assets.append(f"{r.status} {r.url}") if r.status >= 400 and "/api/" not in r.url else None)

        def shot(name):
            if shots:
                page.screenshot(path=str(shots / f"{name}.png"), full_page=True)

        # --- every page renders, CSS applied, no JS errors
        for path in ["/", "/classify", "/playground", "/batch", "/models", "/evaluation", "/errors", "/health-ui", "/demo", "/about"]:
            r = page.goto(a.url + path, wait_until="networkidle")
            check(f"page {path} loads", r.status == 200)
            bg = page.evaluate("getComputedStyle(document.body).fontFamily")
            check(f"css applied {path}", "system-ui" in bg or "sans" in bg.lower(), bg[:40])
            shot(path.strip("/").replace("/", "_") or "dashboard")
        # --- classify
        page.goto(a.url + "/classify")
        page.click("button.chip >> nth=-1")
        filled = page.input_value("#text")
        check("quick example fills textarea", len(filled) > 3, filled[:40])
        page.fill("#text", "Mera card block ho gaya hai")
        avail = page.eval_on_selector_all("#model option:not([disabled])", "els => els.map(e => e.value)")
        for model in avail:
            page.select_option("#model", model)
            page.click("#analyze-btn")
            page.wait_for_selector("#result:not([hidden])")
            page.wait_for_function("document.querySelector('#analyze-btn').disabled === false")
            txt = page.inner_text("#result-body").lower()  # innerText applies CSS text-transform
            check(f"classify result ({model})", all(k in txt for k in ["intent", "urgency", "escalation probability", "route", "inference time"]), txt[:60].replace("\n", " "))
            check(f"escalation meter ({model})", page.locator("#result-body .meter span.on, #result-body .meter span").count() == 10)
        shot("classify_result")
        page.fill("#text", "   ")
        page.click("#analyze-btn")
        page.wait_for_selector(".toast")
        check("empty input -> friendly toast", "non-empty" in page.inner_text("#toasts"))
        # --- playground compare
        page.goto(a.url + "/playground")
        page.fill("#text", "Mera refund abhi tak nahi aaya")
        page.click("#compare-btn")
        page.wait_for_selector("#compare-out section")
        page.wait_for_function("document.querySelector('#compare-btn').disabled === false")
        n = page.locator("#compare-out section").count()
        check("playground renders one card per model", n == 2, f"{n} cards")
        check("playground shows latency", "ms" in page.inner_text("#compare-out"))
        shot("playground_result")
        # --- batch
        csv = "id,text\nr1,Mera card block ho gaya hai\nr2,\nr3,My refund has not arrived\n"
        tmp = Path(tempfile.mkdtemp()) / "in.csv"
        tmp.write_text(csv, encoding="utf-8")
        page.goto(a.url + "/batch")
        page.set_input_files("#file", str(tmp))
        page.click("#batch-btn")
        page.wait_for_selector("#batch-result:not([hidden])")
        summ = page.inner_text("#batch-summary").lower()
        check("batch summary counts", "3" in summ and "failed rows" in summ, summ.replace("\n", " ")[:90])
        check("batch failed row reported", "ID r2" in page.inner_text("#batch-failed"))
        check("batch table rows", page.locator("#batch-rows tr").count() == 2)
        with page.expect_download() as dl:
            page.click("#download-btn")
        content = Path(dl.value.path()).read_text(encoding="utf-8")
        check("batch download is real CSV", content.startswith("id,text,language,intent,urgency,escalation_probability,route") and "r1" in content)
        shot("batch_result")
        bad = Path(tempfile.mkdtemp()) / "bad.csv"
        bad.write_text("foo,bar\n1,2\n")
        page.set_input_files("#file", str(bad))
        page.click("#batch-btn")
        page.wait_for_selector(".toast.error, .toast.warn")
        check("malformed CSV -> toast with reason", "'text' column" in page.inner_text("#toasts"))
        # --- errors table
        page.goto(a.url + "/errors")
        page.wait_for_selector("#err-rows tr")
        total_before = page.inner_text("#err-count")
        check("error table loads real rows", "failed examples" in total_before, total_before)
        page.fill("#q", "card")
        page.wait_for_timeout(700)
        check("error table search filters", page.inner_text("#err-count") != total_before, page.inner_text("#err-count"))
        shot("errors")
        # --- theme
        page.goto(a.url + "/")
        page.click("#theme-btn")
        t1 = page.evaluate("document.documentElement.getAttribute('data-theme')")
        page.reload()
        t2 = page.evaluate("document.documentElement.getAttribute('data-theme')")
        check("theme toggle persists", t1 in ("dark", "light") and t1 == t2, f"{t1}/{t2}")
        # --- mobile
        m = b.new_context(viewport={"width": 390, "height": 800}).new_page()
        for path in ["/", "/classify", "/models", "/evaluation", "/errors", "/health-ui"]:
            m.goto(a.url + path, wait_until="networkidle")
            sw = m.evaluate("document.documentElement.scrollWidth")
            check(f"mobile no horizontal page overflow {path}", sw <= 392, f"scrollWidth={sw}")
        m.goto(a.url + "/")
        check("mobile sidebar hidden by default", not m.evaluate("document.getElementById('sidebar').getBoundingClientRect().right > 10"))
        m.click("#menu-btn")
        m.wait_for_timeout(400)
        check("mobile menu opens", m.evaluate("document.getElementById('sidebar').getBoundingClientRect().right > 100"))
        if shots:
            m.screenshot(path=str(shots / "mobile_menu.png"))
        check("no JS console errors", not errors, "; ".join(errors[:3]))
        check("no 4xx/5xx static or page responses", not failed_assets, "; ".join(failed_assets[:3]))
        b.close()
    bad = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
