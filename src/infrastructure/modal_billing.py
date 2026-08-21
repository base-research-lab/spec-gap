"""Pure helpers for authoritative Modal billing reconciliation.

Modal billing reports are authoritative for metered resource cost.  The
per-model-turn records in :mod:`src.infrastructure.modal_costs` remain useful
for token- and trajectory-level diagnostics, but they are estimates and do not
include the complete lifetime of an App container.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from decimal import Decimal
from typing import Any, Iterable


BILLING_RECONCILIATION_SCHEMA_VERSION = (
    "spec_gap.modal_billing_reconciliation.v2"
)
BASE_BILLING_TAGS = {
    "project": "spec-gap",
    "component": "qwen3-inference",
    "scenario": "scenario1",
}
ANALYSIS_TIERS = frozenset({"exploratory", "definitive"})
_MODAL_TAG_MAX_LENGTH = 63
_MODAL_TAG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _modal_tag_value(value: str) -> str:
    """Return a Modal-legal tag value, hashing when the raw value is too long."""

    safe = _MODAL_TAG_UNSAFE.sub("-", value).strip("-_.")
    if not safe:
        raise ValueError("billing tag value is empty after sanitizing")
    if len(safe) <= _MODAL_TAG_MAX_LENGTH:
        return safe
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    keep = _MODAL_TAG_MAX_LENGTH - len(digest) - 1
    return f"{safe[:keep].rstrip('-_.')}-{digest}"


def as_decimal(value: Any, field: str) -> Decimal:
    """Return one finite, non-negative billing value as a Decimal."""

    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:  # pragma: no cover - Decimal has varied errors
        raise ValueError(f"{field} must be decimal-compatible") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def decimal_text(value: Decimal) -> str:
    """Serialize a Decimal without converting authoritative money to float."""

    return format(value, "f")


def validate_analysis_tier(value: str) -> str:
    """Return one explicit paper-analysis tier for a Modal run."""

    if not isinstance(value, str) or value.strip() not in ANALYSIS_TIERS:
        raise ValueError(
            "analysis_tier must be one of "
            f"{sorted(ANALYSIS_TIERS)}"
        )
    return value.strip()


def scenario1_billing_tags(
    *,
    domain_ids: Iterable[str],
    generation_protocol_ids: Iterable[str],
    run_kind: str,
    analysis_tier: str,
) -> dict[str, str]:
    """Build stable Modal App tags for future cost attribution."""

    domains = sorted({str(item).strip() for item in domain_ids if str(item).strip()})
    protocols = sorted({
        str(item).strip()
        for item in generation_protocol_ids
        if str(item).strip()
    })
    if not domains:
        raise ValueError("at least one domain_id is required for billing tags")
    if not protocols:
        raise ValueError(
            "at least one generation_protocol_id is required for billing tags"
        )
    if not isinstance(run_kind, str) or not run_kind.strip():
        raise ValueError("run_kind must be a non-empty string")
    tier = validate_analysis_tier(analysis_tier)
    return {
        **BASE_BILLING_TAGS,
        "domains": _modal_tag_value(".".join(domains)),
        "generation_protocols": _modal_tag_value(".".join(protocols)),
        "run_kind": _modal_tag_value(run_kind.strip()),
        "analysis_tier": tier,
    }


def aggregate_billing_rows(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate normalized hourly Modal rows into App and project totals."""

    by_app: dict[str, dict[str, Any]] = {}
    project_resources: defaultdict[str, Decimal] = defaultdict(Decimal)
    hourly_row_count = 0

    for row in rows:
        hourly_row_count += 1
        object_id = row.get("object_id")
        if not isinstance(object_id, str) or not object_id.startswith("ap-"):
            raise ValueError("billing row object_id must be a Modal App id")
        cost = as_decimal(row.get("cost"), "billing row cost")
        resources = row.get("cost_by_resource")
        if not isinstance(resources, dict):
            raise ValueError("billing row cost_by_resource must be an object")
        normalized_resources = {
            str(name): as_decimal(value, f"resource {name}")
            for name, value in resources.items()
        }
        interval = str(row.get("interval_start") or "")
        if not interval:
            raise ValueError("billing row interval_start is required")

        current = by_app.setdefault(object_id, {
            "modal_app_id": object_id,
            "description": row.get("description"),
            "environment": row.get("environment"),
            "first_interval_start": interval,
            "last_interval_start": interval,
            "metered_cost_usd": Decimal(0),
            "cost_by_resource_usd": defaultdict(Decimal),
            "tags": dict(row.get("tags") or {}),
            "hourly_row_count": 0,
        })
        # Mid-run metadata changes, including retagging, are integrity errors
        # by design.
        if current["description"] != row.get("description"):
            raise ValueError(f"description changed within App {object_id}")
        if current["environment"] != row.get("environment"):
            raise ValueError(f"environment changed within App {object_id}")
        if current["tags"] != dict(row.get("tags") or {}):
            raise ValueError(f"tags changed within App {object_id}")
        current["first_interval_start"] = min(
            current["first_interval_start"], interval
        )
        current["last_interval_start"] = max(
            current["last_interval_start"], interval
        )
        current["metered_cost_usd"] += cost
        current["hourly_row_count"] += 1
        for name, value in normalized_resources.items():
            current["cost_by_resource_usd"][name] += value
            project_resources[name] += value

    apps = []
    for current in by_app.values():
        apps.append({
            **current,
            "metered_cost_usd": decimal_text(current["metered_cost_usd"]),
            "cost_by_resource_usd": {
                name: decimal_text(value)
                for name, value in sorted(
                    current["cost_by_resource_usd"].items()
                )
            },
        })
    apps.sort(key=lambda item: (
        item["first_interval_start"], item["modal_app_id"]
    ))
    total = sum(
        (as_decimal(item["metered_cost_usd"], "App cost") for item in apps),
        Decimal(0),
    )
    summary = {
        "modal_app_count": len(apps),
        "hourly_billing_row_count": hourly_row_count,
        "metered_cost_usd": decimal_text(total),
        "cost_by_resource_usd": {
            name: decimal_text(value)
            for name, value in sorted(project_resources.items())
        },
    }
    return apps, summary
