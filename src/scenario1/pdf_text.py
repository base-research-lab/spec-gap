"""Extract UTF-8 text from source PDFs at retrieval time."""

from __future__ import annotations

import functools
import shutil
import subprocess
import tempfile
from pathlib import Path

_FALLBACK_PDFTOTEXT = Path("/Users/oudoum/opt/anaconda3/bin/pdftotext")


def pdftotext_executable() -> str:
    """Return the pdftotext binary used for on-the-fly PDF extraction."""

    found = shutil.which("pdftotext")
    if found:
        return found
    if _FALLBACK_PDFTOTEXT.is_file():
        return str(_FALLBACK_PDFTOTEXT)
    raise RuntimeError("pdftotext is required (Poppler) to convert PDFs to text")


@functools.lru_cache(maxsize=128)
def extract_pdf_text(path: str) -> str:
    """Run ``pdftotext -raw`` on one PDF and return the extracted UTF-8 text."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing source PDF {source}")
    executable = pdftotext_executable()
    with tempfile.TemporaryDirectory(prefix="specgap-pdftotext-") as temp_dir:
        extracted = Path(temp_dir) / "extracted.txt"
        subprocess.run(
            [executable, "-raw", str(source), str(extracted)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return extracted.read_text(encoding="utf-8")
