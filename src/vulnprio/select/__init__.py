"""Resource-constrained remediation selection (DESIGN.md 3.7, Gap 10).

Turns a ranking into a plan: a 0/1 knapsack over remediation hours that maximises captured
``chain_adjusted_loss``, charging each root cause once.
"""

from __future__ import annotations

from vulnprio.select.knapsack import (
    MIN_HOURS,
    Cluster,
    SelectionItem,
    cluster_items,
    items_from_enriched,
    items_from_ranking,
    select_under_budget,
)

__all__ = [
    "MIN_HOURS",
    "SelectionItem",
    "Cluster",
    "cluster_items",
    "items_from_enriched",
    "items_from_ranking",
    "select_under_budget",
]
