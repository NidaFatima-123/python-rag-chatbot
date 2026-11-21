"""Optional: download the official Python tutorial (docs.python.org) into data/official_docs/.

Run from the project root:   python scripts/download_docs.py
Then click "Rebuild knowledge base" in the app.
Needs internet access. If a page fails, it is skipped and the starter files still work.
"""
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://docs.python.org/3/tutorial/"
PAGES = [
    "appetite", "interpreter", "introduction", "controlflow", "datastructures",
    "modules", "inputoutput", "errors", "classes", "stdlib", "stdlib2", "venv",
]
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "official_docs"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for page in PAGES:
        url = f"{BASE_URL}{page}.html"
        try:
            response = requests.get(url, timeout=30,
                                    headers={"User-Agent": "python-rag-tutor/1.0 (learning project)"})
            response.raise_for_status()
        except Exception as error:
            print(f"[skip] {url} -> {error}")
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        main_div = soup.find("div", attrs={"role": "main"}) or soup.body
        text = main_div.get_text("\n").replace("\u00b6", "")
        lines = [ln.rstrip() for ln in text.splitlines()]
        text = "\n".join(lines).strip()
        header = f"Source: {url} (Python 3 official tutorial)\n\n"
        (OUT_DIR / f"py_tutorial_{page}.txt").write_text(header + text, encoding="utf-8")
        print(f"[ok] {page}.html")
        saved += 1
        time.sleep(1)  # be polite to the server
    print(f"Done. {saved}/{len(PAGES)} pages saved to {OUT_DIR}")
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
