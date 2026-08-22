"""Per-domain 3 style x 3 injection-position grid for fellow_packages_New."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

KNOWN_DOMAINS = (
    "aihc",
    "telecoms",
    "policy",
    "petro",
    "neuro",
    "macro",
    "kg",
    "fin",
    "convex",
)
POSITIONS = ("begin", "middle", "end")
STYLE_KEYS = ("12", "20", "28")
STYLE_IDS = {
    "12": "12_docid_in_calibration_line",
    "20": "20_full_note_self_replicate",
    "28": "28_tool_in_traveling_line",
}
# Grid positions → Ifeoluwa injected-twin labels in fellow_packages_New registries.
POSITION_PDF_LABELS = {
    "begin": "beginning",
    "middle": "middle",
    "end": "before_references",
}


def injected_pdf_for_position(source: dict[str, Any], position: str) -> str:
    """Return the carrier injected-PDF filename for one grid position."""

    if position not in POSITION_PDF_LABELS:
        raise ValueError(
            f"unknown grid position {position!r}; known: "
            + ", ".join(POSITIONS)
        )
    label = POSITION_PDF_LABELS[position]
    injection = source.get("injection") or {}
    named = injection.get("injected_pdfs")
    if isinstance(named, dict):
        filename = named.get(label)
        if isinstance(filename, str) and filename.lower().endswith(".pdf"):
            return filename
    for variant in injection.get("injected_variants") or []:
        if not isinstance(variant, dict):
            continue
        if variant.get("label") != label:
            continue
        filename = variant.get("injected_pdf")
        if isinstance(filename, str) and filename.lower().endswith(".pdf"):
            return filename
    raise ValueError(
        f"no injected PDF registered for position {position!r} ({label})"
    )


def injected_pdf_for_cell(
    source: dict[str, Any], style_key: str, position: str
) -> str:
    """Return the carrier injected-PDF filename for one style x position cell.

    Since styles 12/20/28 now carry distinct payload text (see
    ``build_payload`` in the grid-build script), each style needs its own
    injected-PDF twin per position rather than sharing one PDF across all
    three styles. Registries record that as
    ``injection.injected_pdfs.by_style.<style_key>.<label>``. Falls back to
    the position-only (shared) PDF via ``injected_pdf_for_position`` when a
    domain hasn't registered style-specific twins yet, so unmigrated
    domains keep working exactly as before.
    """

    if style_key not in STYLE_KEYS:
        raise ValueError(
            f"unknown style key {style_key!r}; known: " + ", ".join(STYLE_KEYS)
        )
    if position not in POSITION_PDF_LABELS:
        raise ValueError(
            f"unknown grid position {position!r}; known: "
            + ", ".join(POSITIONS)
        )
    label = POSITION_PDF_LABELS[position]
    injection = source.get("injection") or {}
    named = injection.get("injected_pdfs")
    if isinstance(named, dict):
        by_style = named.get("by_style")
        if isinstance(by_style, dict):
            style_entry = by_style.get(style_key)
            if isinstance(style_entry, dict):
                filename = style_entry.get(label)
                if isinstance(filename, str) and filename.lower().endswith(".pdf"):
                    return filename
    return injected_pdf_for_position(source, position)


def parse_domains(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated domain list, or ``all`` for every known folder."""

    text = (raw or "").strip()
    if not text:
        raise ValueError(
            "specify domains, for example 'macro' or 'macro,fin' or 'all'"
        )
    if text.lower() == "all":
        return KNOWN_DOMAINS
    folders = tuple(item.strip() for item in text.split(",") if item.strip())
    if not folders:
        raise ValueError(
            "specify domains, for example 'macro' or 'macro,fin' or 'all'"
        )
    unknown = [folder for folder in folders if folder not in KNOWN_DOMAINS]
    if unknown:
        raise ValueError(
            "unknown domain(s) "
            + ", ".join(unknown)
            + "; known: "
            + ", ".join(KNOWN_DOMAINS)
        )
    return tuple(dict.fromkeys(folders))


def expand_domain_grid(domains: Iterable[str]) -> list[dict[str, str]]:
    """Return the 9 style x position cells for each requested domain."""

    folders = tuple(domains)
    unknown = [folder for folder in folders if folder not in KNOWN_DOMAINS]
    if unknown:
        raise ValueError(
            "unknown domain(s) "
            + ", ".join(unknown)
            + "; known: "
            + ", ".join(KNOWN_DOMAINS)
        )
    cells: list[dict[str, str]] = []
    for folder in folders:
        for style_key in STYLE_KEYS:
            for position in POSITIONS:
                style_id = STYLE_IDS[style_key]
                cells.append({
                    "folder": folder,
                    "position": position,
                    "style_key": style_key,
                    "style_id": style_id,
                    "domain_id": f"{folder}_newgrid_{style_key}_{position}",
                })
    return cells


def cell_registry_path(inputs_root: Path, cell: dict[str, str]) -> Path:
    """Return the domain_config.json path for one style x position cell."""

    return (
        inputs_root
        / "fellow_packages_New"
        / cell["folder"]
        / "attack_styles"
        / cell["style_id"]
        / cell["position"]
        / "domain_config.json"
    )


def registry_paths_for_domains(
    domains: Iterable[str],
    *,
    inputs_root: Path,
) -> list[Path]:
    """Return registry paths for every 9-cell expansion of the given domains."""

    return [
        cell_registry_path(inputs_root, cell)
        for cell in expand_domain_grid(domains)
    ]
