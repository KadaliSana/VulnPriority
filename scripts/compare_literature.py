"""Benchmark VulnPriority against the competitors re-implemented from the literature review.

Reloads a finished run (scans, enriched findings, attack-graph scores and labels), then puts
LambdaMART, the seven practitioner baselines and the eight literature competitors through
the one evaluation harness: same time-ordered folds, same confirmed-exploitation labels,
same metrics. Nothing is re-generated, so the numbers are directly comparable with the run's
own report.

    python scripts/compare_literature.py runs-lit/<run> [inclusive|strict]

Writes ``literature_comparison_<protocol>.json`` and ``.md`` into the run directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from vulnpriority.core.config import PipelineConfig
from vulnpriority.core.enums import RankerName
from vulnpriority.eval.report import _aggregate_bundles
from vulnpriority.pipeline.artifacts import RunArtifacts
from vulnpriority.pipeline.stages import evaluate_stage
from vulnpriority.rank.literature import LITERATURE_RANKERS

METRICS = ("ndcg@5", "ndcg@10", "kendall_tau_vs_cvss", "precision@10", "risk_capture@10", "map", "mrr", "mean_rank_of_exploited",
           "pr_auc", "mcc", "f1_minority", "coverage")


def _current_config(raw: dict) -> PipelineConfig:
    """The run's config, minus keys a newer schema has retired (e.g. a moved field)."""
    while True:
        try:
            return PipelineConfig.model_validate(raw)
        except ValidationError as err:
            stale = [e["loc"] for e in err.errors() if e["type"] == "extra_forbidden"]
            if not stale:
                raise
            for loc in stale:
                node = raw
                for part in loc[:-1]:
                    node = node[part]
                node.pop(loc[-1], None)


def main(run_dir: str, protocol: str = "inclusive") -> None:
    """``inclusive`` grades against every confirmed-exploitation label, KEV included (the
    paper's protocol); ``strict`` demotes positives that only KEV or exploit-feed evidence
    supports, so a ranker earns nothing for reading the answer off its own columns."""
    root = Path(run_dir)
    raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
    raw.setdefault("evaluation", {})["exclude_circular_labels"] = protocol == "strict"
    config = _current_config(raw)
    (root / "config.json").rename(root / "config.json.orig")
    try:
        art = RunArtifacts.load(root)
    finally:
        (root / "config.json.orig").rename(root / "config.json")
    rankers = [RankerName.LAMBDAMART, *config.evaluation.baselines, *LITERATURE_RANKERS]
    bundles = evaluate_stage(config, art.scans, art.enriched, art.labels,
                             chain_scores=art.chain_scores, flags=config.flags(), rankers=rankers)
    summary = _aggregate_bundles(bundles)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for name in rankers:
        entry = summary.get(name.value, {}).get("values", {})
        out[name.value] = {
            key: {k: entry[key][k] for k in ("value", "ci_low", "ci_high") if k in entry[key]}
            for key in METRICS if key in entry
        }
    (root / f"literature_comparison_{protocol}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = ["| ranker | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for name, metrics in sorted(out.items(), key=lambda kv: -kv[1].get("ndcg@10", {}).get("value", 0)):
        cells = [f"{metrics[m]['value']:.3f}" if m in metrics else "-" for m in METRICS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (root / f"literature_comparison_{protocol}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "inclusive")
