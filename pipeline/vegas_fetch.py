"""Historical MLB preseason win totals (yardstick only, never a model input; user OK 2026-11-24).
Source: sportsoddshistory.com (now covers.com/sportsoddshistory). Runs in Actions (sandbox can't reach it).
usage: python pipeline/vegas_fetch.py Y0 Y1  -> data/vegas/raw_<y>.html, data/vegas/win_totals.parquet
"""
import io
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "vegas"
URL = "https://www.covers.com/sportsoddshistory/mlb-win/?y={y}&sa=mlb&t=win"


def render(urls):
    from playwright.sync_api import sync_playwright
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36")
        for k, u in urls.items():
            pg.goto(u, wait_until="networkidle", timeout=90000)
            try:
                pg.wait_for_selector("table", timeout=20000)
            except Exception:
                pass
            out[k] = pg.content()
        b.close()
    return out


def main(y0, y1):
    OUT.mkdir(parents=True, exist_ok=True)
    pages = render({y: URL.format(y=y) for y in range(y0, y1 + 1)})
    rows = []
    for y, html in pages.items():
        (OUT / f"raw_{y}.html").write_text(html)
        try:
            tabs = pd.read_html(io.StringIO(html))
        except ValueError:
            print(y, "no tables")
            continue
        for i, t in enumerate(tabs):
            t.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
            print(y, i, t.shape, list(t.columns)[:8])
            rows.append(t.astype(str).assign(season=y, table=i))
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(OUT / "win_totals_raw.parquet", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
