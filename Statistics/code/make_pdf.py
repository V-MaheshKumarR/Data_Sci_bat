"""Exports Statistics_Beginner_Slides.html to PDF using DeckTape.

DeckTape drives a real headless Chromium through reveal.js, so the PDF is
pixel-identical to the browser (fonts, KaTeX, backgrounds, images).

Requires Node.js. Run:  python make_pdf.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DECK = BASE / "Statistics_Beginner_Slides.html"
PDF = BASE / "Statistics_Beginner_Slides.pdf"

if not shutil.which("npx"):
    sys.exit("Node.js not found. Install it from https://nodejs.org, then re-run.")

subprocess.run(
    ["npx", "-y", "decktape@3", "reveal", DECK.resolve().as_uri(), str(PDF),
     "--size", "1280x960",
     "--load-pause", "1500",   # let the CDN CSS/JS and KaTeX settle
     "--pause", "400"],        # per-slide settle time
    check=True,
    shell=True,
)

print(f"\nwrote {PDF.name}  ({PDF.stat().st_size / 1024:,.0f} KB)")
