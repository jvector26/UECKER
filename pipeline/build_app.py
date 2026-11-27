"""Build the UECKER app into site/ from the templates in site_src/ (never edit site/index.html or site/sw.js by hand).
usage: python pipeline/build_app.py      (after pipeline/export_app.py has written site/data/*.json)
Writes: site/index.html (CSS + JS inlined), site/sw.js (cache version = hash of shell + data), manifest, icons.
"""
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, SITE = ROOT / "site_src", ROOT / "site"


def icons(out):
    from PIL import Image, ImageDraw, ImageFont
    out.mkdir(parents=True, exist_ok=True)
    for size, name in ((512, "icon-512.png"), (192, "icon-192.png"), (180, "apple-touch-icon.png")):
        s = size / 512
        im = Image.new("RGB", (size, size), "#1D4C9E")
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([60 * s, 60 * s, 452 * s, 452 * s], radius=28 * s, fill="#5B3A22")   # wood frame
        d.rectangle([92 * s, 92 * s, 420 * s, 420 * s], fill="#F4C430")
        d.ellipse([140 * s, 140 * s, 372 * s, 372 * s], fill="#FBF6E8", outline="#17213A", width=max(2, int(10 * s)))
        for x0 in (175, 337):  # stitches
            d.arc([x0 * s - 70 * s, 160 * s, x0 * s + 70 * s, 352 * s], 290 if x0 < 256 else 110, 70 if x0 < 256 else 250,
                  fill="#C8302C", width=max(2, int(8 * s)))
        try:
            f = ImageFont.truetype("DejaVuSerif-Bold.ttf", int(150 * s))
        except OSError:
            f = ImageFont.load_default()
        d.text((256 * s, 262 * s), "U", font=f, fill="#17213A", anchor="mm")
        im.save(out / name)


def main():
    SITE.mkdir(exist_ok=True)
    css, js = (SRC / "app.css").read_text(), (SRC / "app.js").read_text()
    html = (SRC / "index.html").read_text().replace("/*__CSS__*/", css).replace("/*__JS__*/", js)
    h = hashlib.sha1(html.encode())
    for f in sorted((SITE / "data").glob("*.json")):
        h.update(f.read_bytes())
    V = h.hexdigest()[:10]
    (SITE / "index.html").write_text(html)
    (SITE / "sw.js").write_text((SRC / "sw.js").read_text().replace("__V__", V))
    shutil.copy(SRC / "manifest.webmanifest", SITE / "manifest.webmanifest")
    (SITE / ".nojekyll").write_text("")
    if not (SITE / "icons" / "icon-512.png").exists():
        icons(SITE / "icons")
    print("built site/ version", V)


if __name__ == "__main__":
    main()
