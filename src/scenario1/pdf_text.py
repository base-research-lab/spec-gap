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


def detect_single_insertion(clean_text: str, injected_text: str) -> tuple[int, str]:
    """Return the offset and exact delta when injected = clean + one insertion."""

    if clean_text == injected_text:
        raise ValueError("clean and injected carrier texts are identical")
    prefix_length = 0
    for clean_char, injected_char in zip(clean_text, injected_text):
        if clean_char != injected_char:
            break
        prefix_length += 1
    clean_suffix = clean_text[prefix_length:]
    if clean_suffix and not injected_text.endswith(clean_suffix):
        raise ValueError("carrier PDFs differ beyond one contiguous insertion")
    injected_end = (
        len(injected_text) - len(clean_suffix) if clean_suffix else len(injected_text)
    )
    insertion_delta = injected_text[prefix_length:injected_end]
    if not insertion_delta:
        raise ValueError("the detected insertion delta is empty")
    if injected_text[:prefix_length] + injected_text[injected_end:] != clean_text:
        raise ValueError("removing the detected delta does not recover clean text")
    return prefix_length, insertion_delta
