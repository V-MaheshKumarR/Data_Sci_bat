"""Exports DT_SVM_Slides.html to PDF using DeckTape (needs Node.js).

Run:  python make_pdf.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DECK = BASE / "DT_SVM_Slides.html"
PDF = BASE / "DT_SVM_Slides.pdf"

if not shutil.which("npx"):
    sys.exit("Node.js not found. Install it from https://nodejs.org, then re-run.")

subprocess.run(
    ["npx", "-y", "decktape@3", "reveal", DECK.resolve().as_uri(), str(PDF),
     "--size", "1280x960", "--load-pause", "1500", "--pause", "400"],
    check=True, shell=True,
)

print(f"\nwrote {PDF.name}  ({PDF.stat().st_size / 1024:,.0f} KB)")
