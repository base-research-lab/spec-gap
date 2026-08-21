"""Review controls for the active Knowledge Graphs package.

Former filename: ``tests/test_kg_v2_package.py``.
Original date: not encoded in the former filename.
Renamed on: 2026-08-10.
Purpose: spell out the domain and avoid a protocol revision in the test name.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from src.scenario1 import generator


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "experiments" / "scenario1" / "inputs"
KG_ROOT = INPUTS / "fellow_packages" / "kg"
KG_REGISTRY_PATH = KG_ROOT / "domain_config.json"
MACRO_REGISTRY_PATH = (
    INPUTS / "fellow_packages" / "macro" / "domain_config.json"
)

WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z-]*\b")
CAPITALIZED_SPAN_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z-]*|[A-Z]{2,})"
    r"(?:[ \t]+(?:[A-Z][A-Za-z-]*|[A-Z]{2,})){1,}\b"
)
PLANNER_FIXTURE = "Retrieve and summarize the selected evidence."


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _worker_message_content(registry_path: Path, treatment: str) -> str:
    registry = generator.load_registry(registry_path)
    record = generator.build_record(registry, "2-hop", treatment)
    worker = next(
        event
        for event in record["trajectory_trace"]["full_events"]
        if event.get("agent_id") == "worker_1"
    )
    request = generator.build_generation_request(
        record,
        worker,
        thinking_mode="off",
        upstream_message=PLANNER_FIXTURE,
    )
    return "\n".join(message["content"] for message in request["messages"])


def _density_metrics(text: str) -> dict[str, int | float | str]:
    words = WORD_PATTERN.findall(text)
    spans = CAPITALIZED_SPAN_PATTERN.findall(text)
    return {
        "message_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "characters": len(text),
        "word_tokens": len(words),
        "capitalized_multi_token_spans": len(spans),
        "spans_per_1000_words": round(1000 * len(spans) / len(words), 6),
    }


def test_kg_payload_is_classified_from_the_pinned_qwen_tokenizer_audit():
    registry = _load_json(KG_REGISTRY_PATH)
    plan = _load_json(
        KG_ROOT / "retrieval" / "plan.json"
    )
    preflight = _load_json(
        KG_ROOT / "retrieval" / "context_check.json"
    )
    audit = _load_json(INPUTS / registry["provenance"]["qwen_special_token_audit"])
    payload = registry["injection"]["wordings"][registry["assigned_wording"]]

    assert hashlib.sha256(payload.encode()).hexdigest() == audit["payload"]["sha256"]
    assert len(payload) == audit["payload"]["characters"] == 372
    assert audit["payload"]["token_count"] == 68
    assert audit["tokenizer_json_sha256"] == plan["tokenizer"][
        "tokenizer_json_sha256"
    ]
    assert audit["tokenizer_json_sha256"] == preflight["tokenizer_json_sha256"]
    assert {
        item["text"]: item["token_id"] for item in audit["embedded_markers"]
    } == {"<|im_start|>": 151644, "<|im_end|>": 151645}
    assert all(marker in payload for marker in ("<|im_start|>", "<|im_end|>"))
    assert audit["apply_chat_template_probe"]["embedded_pair_delta"] == {
        "im_start_token_count": 1,
        "im_end_token_count": 1,
    }

    counts = {
        (case["treatment"], case["thinking_mode"]): case["input_token_count"]
        for case in preflight["cases"]
    }
    assert counts[("injected", "off")] - counts[("clean", "off")] == 69
    assert counts[("injected", "on")] - counts[("clean", "on")] == 69


@pytest.mark.parametrize(
    ("cell", "registry_path", "treatment"),
    [
        ("kg_clean", KG_REGISTRY_PATH, "clean"),
        ("kg_injected", KG_REGISTRY_PATH, "injected"),
        ("macro_clean", MACRO_REGISTRY_PATH, "clean"),
        ("macro_injected", MACRO_REGISTRY_PATH, "injected"),
    ],
)
def test_kg_entity_density_proxy_is_reproducible(
    cell: str,
    registry_path: Path,
    treatment: str,
):
    registry = _load_json(KG_REGISTRY_PATH)
    audit = _load_json(INPUTS / registry["provenance"]["entity_density_audit"])
    text = _worker_message_content(registry_path, treatment)

    assert _density_metrics(text) == audit["cells"][cell]


def test_kg_clean_entity_proxy_is_not_above_the_plain_text_comparator():
    registry = _load_json(KG_REGISTRY_PATH)
    audit = _load_json(INPUTS / registry["provenance"]["entity_density_audit"])
    kg_rate = audit["cells"]["kg_clean"]["spans_per_1000_words"]
    macro_rate = audit["cells"]["macro_clean"]["spans_per_1000_words"]

    assert kg_rate < macro_rate
    assert audit["clean_comparison"] == {
        "kg_minus_macro_spans_per_1000_words": round(kg_rate - macro_rate, 6),
        "kg_to_macro_rate_ratio": round(kg_rate / macro_rate, 6),
        "kg_percent_difference": round(100 * (kg_rate / macro_rate - 1), 3),
    }
