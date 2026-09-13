"""Build and audit the corpus the shipped ranker is trained on.

Why this exists at all is worth stating, because "use a real dataset" is the obvious first
answer and it does not work. The learned ranker needs three things at once:

1. findings grouped into *scans*, because a pairwise ranking objective compares within a
   query group and a single group teaches it nothing;
2. per-finding features spanning asset criticality, attacker likelihood and attack-graph
   position - properties of an application *deployment*, not of a CVE;
3. per-finding confirmed exploitation, plus the counterfactual "would this have been
   exploited had we not fixed it", which no observational corpus can contain because only
   one arm is ever observed.

Every public dataset supplies at most the CVE layer: NVD, CISA KEV, EPSS, Exploit-DB,
CVEfixes. Those are already consumed by this framework as *feature* inputs. None of them
carries scan grouping, deployment context or outcomes - and on a real web application scan
most findings are not CVE-bearing at all, so CVE-keyed labels would not even cover the
population being ordered.

So the corpus is synthetic, and the burden that places on it is the point of the audit
below: a synthetic corpus is only worth training on if the task it poses is *hard in the
same way the real task is hard*. The checks are written to fail loudly rather than to pass
quietly.

**Built in shards.** Ten thousand scans is roughly 386,000 findings, and an enriched
finding is a deep tree of Pydantic models. Holding the whole corpus live in order to build
one feature matrix at the end exhausted the machine and took the desktop down with it. Each
shard is therefore generated, enriched, labelled and reduced to its numeric matrix on its
own, and only that matrix - a few megabytes of float32 - survives into the next shard. Peak
memory is one shard's worth however large the total, and a shard already on disk is reused,
so an interrupted build resumes instead of starting over.

    python scripts/build_training_corpus.py --scans 10000
    python scripts/build_training_corpus.py --scans 10000 --shard-scans 250   # less memory
    python scripts/build_training_corpus.py --scans 500                       # a smoke run
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vulnpriority.core.config import SyntheticConfig, load_config
from vulnpriority.core.models import FeatureFrame
from vulnpriority.pipeline.runner import PipelineRunner
from vulnpriority.pipeline.stages import build_ranker, feature_stage, scoring_labels
from vulnpriority.synth.feeds import write_feed_fixtures
from vulnpriority.synth.generator import SyntheticDataset
from vulnpriority.synth.intel import attach_intel

PROJECT = Path(__file__).resolve().parents[1]
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Quality criteria
#
# Each is a property the corpus must have for a model trained on it to be worth shipping,
# with a threshold chosen before the numbers were seen.
# ---------------------------------------------------------------------------

#: Positives must stay rare. Exploitation in the field is a low-single-digit-percent event,
#: and a corpus that balances the classes would train a model calibrated for a world that
#: does not exist.
MAX_POSITIVE_RATE = 0.15
MIN_POSITIVE_RATE = 0.01

#: Every feature the model is given must vary. A constant column is a column the model
#: cannot learn from and a reader cannot interpret.
MIN_FEATURE_VARIANCE = 1e-12

#: No single observable may be a near-perfect stand-in for the label. If one is, the corpus
#: is teaching "sort by that column" and the learned ranker's advantage is an illusion that
#: will not survive contact with a real scan.
MAX_SINGLE_FEATURE_AUC = 0.95

#: The learned ranker has to beat the best single-signal baseline by enough to be worth the
#: machinery. Below this the honest answer is to ship the baseline.
MIN_LIFT_OVER_BEST_BASELINE = 0.03

#: Query groups must be large enough for a within-scan ordering to mean something.
MIN_MEAN_GROUP_SIZE = 10.0

#: Enough independently-confirmed positives to measure against. KEV membership and exploit
#: evidence are both accepted ground truth and feature columns, so a positive resting on
#: nothing else is a lookup; only the oracle-backed ones can grade a model that reads those
#: columns. Below this count the held-out NDCG is noise however good it looks.
MIN_INDEPENDENT_POSITIVES = 500

#: On-disk shard format. Bump it whenever the feature layer, the label policy or anything
#: else that changes a shard's contents changes, so that a resumed build rebuilds rather
#: than silently mixing two generations of matrix.
SHARD_FORMAT = 3


def ndcg_at_k(order: list[int], k: int) -> float:
    """Exponential-gain NDCG@k over a list of relevance grades already in ranked order."""
    def dcg(grades: list[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))

    ideal = sorted(order, reverse=True)[:k]
    best = dcg(ideal)
    return dcg(order[:k]) / best if best > 0 else 0.0


def mean_ndcg(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
    """Mean NDCG@k across query groups, which is how the protocol aggregates it."""
    out, offset = [], 0
    for size in groups:
        size = int(size)
        block_scores = scores[offset : offset + size]
        block_labels = labels[offset : offset + size]
        offset += size
        if size == 0:
            continue
        order = np.argsort(-block_scores, kind="stable")
        out.append(ndcg_at_k([int(block_labels[i]) for i in order], k))
    return float(np.mean(out)) if out else 0.0


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC by rank statistic. Ties get averaged ranks, so a constant column gives 0.5."""
    positives = labels > 0
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks within ties so a low-cardinality column is not flattered.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------


def build_shard(
    n_scans: int,
    findings_per_app: tuple[int, int],
    seed: int,
    work_dir: Path,
) -> dict[str, Any]:
    """Generate one shard of the world and reduce it to arrays.

    Everything heavy - the latent world, the scans, the enriched findings - is local to this
    call and unreachable once it returns. What comes back is the feature matrix and the few
    columns the audit needs, which is three orders of magnitude smaller.

    Each shard gets its own seed, so the shards are independent draws from the same
    generative process rather than twenty copies of one world.
    """
    scans_per_app = 5
    n_apps = max(1, n_scans // scans_per_app)
    synthetic = SyntheticConfig(
        n_apps=n_apps,
        scans_per_app=scans_per_app,
        findings_per_app=findings_per_app,
        seed=seed,
    )
    base = load_config(PROJECT / "configs" / "offline.yaml")

    started = time.time()
    # ``write=False``: the scan documents run to tens of megabytes a shard and nothing reads
    # them back, because the pipeline runs against the in-memory dataset. Only the feed
    # fixtures have to exist on disk, since ``FeedsConfig.fixture_dir`` is a path.
    dataset = SyntheticDataset.generate(synthetic, write=False)
    fixture_dir = work_dir / "feeds"
    write_feed_fixtures(dataset.world, fixture_dir, pages=dataset.pages)
    summary = dataset.summary()

    feeds = base.feeds.model_copy(update={"fixture_dir": fixture_dir})
    config = base.model_copy(update={"feeds": feeds, "synthetic": synthetic})

    # ``chain`` as well as ``label``: labelling does not depend on the attack graph, so
    # asking only for ``label`` runs a closure that excludes it and leaves every Component C
    # feature at zero. The model would then be blind to attack chains - one of the three
    # components the whole framework is built around - and nothing downstream would say so.
    runner = PipelineRunner(config=config, dataset=dataset, strict=False)
    artifacts = runner.run(config, stages=["chain", "label"], resume=False, save=False)

    # Fold in synthetic retrieved intelligence. Without it the seven ``a_intel_*`` columns
    # are constant across the corpus, a tree cannot split on a constant, and the shipped
    # model would ignore the live intelligence a real run gathers - see synth/intel.py.
    by_cve = dataset.world.index()
    latents = {}
    for latent in dataset.latent_findings:
        vuln = by_cve.get(latent.cve_id) if latent.cve_id else None
        if vuln is not None:
            latents[latent.finding_id] = (vuln.true_exploitability, vuln.true_attacker_interest)
    enriched = attach_intel(artifacts.enriched, latents, seed=seed)
    covered = sum(1 for item in enriched if item.intel_result is not None)

    frame = feature_stage(config, enriched, artifacts.chain_scores)

    # Two label vectors, and the difference between them is the point.
    #
    # ``y_optimistic`` is what the default policy produces: KEV membership and exploit
    # evidence count as confirmed exploitation. Both are also feature columns the ranker
    # reads, so a positive justified by nothing else is telling the model the answer in its
    # own input row. Scoring against those measures a join, not a prediction.
    #
    # ``y`` is what survives ``scoring_labels``: the oracle-backed positives, which no
    # feature column carries. It is the headline, it is what the model is trained on, and
    # the gap between the two is reported rather than quietly enjoyed.
    scoring_set = scoring_labels(config, artifacts.labels)
    honest = scoring_set.relevance()
    optimistic = artifacts.labels.relevance()
    observed = {item.finding_id: item.finding.observed_at for item in enriched}
    impact = {item.finding_id: float(item.impact.total) for item in enriched}

    ids = list(frame.finding_ids)
    return {
        "format": SHARD_FORMAT,
        "seed": seed,
        "flags": config.flags().label(),
        "feature_names": np.array(frame.feature_names),
        "finding_ids": np.array(ids),
        "group_ids": np.array(frame.group_ids),
        "X": frame.X.to_numpy(dtype=np.float32),
        "y": np.array([float(honest.get(f, 0)) for f in ids], dtype=np.float32),
        "y_optimistic": np.array([float(optimistic.get(f, 0)) for f in ids], dtype=np.float32),
        "impact": np.array([impact.get(f, 0.0) for f in ids], dtype=np.float32),
        "observed": np.array(
            [
                (observed[f].replace(tzinfo=timezone.utc) - EPOCH).total_seconds()
                if f in observed else 0.0
                for f in ids
            ],
            dtype=np.float64,
        ),
        "circular_positives": len(artifacts.labels.circular_labels()),
        "scans": int(summary["scans"]),
        "findings": int(summary["findings"]),
        "apps": int(summary["apps"]),
        "endpoints": int(summary["endpoints"]),
        "cves": int(summary.get("cves", 0)),
        "intel_covered": covered,
        "seconds": time.time() - started,
    }


#: Arrays go in the npz; everything else travels beside them as one JSON blob, because npz
#: stores arrays and a scalar round-trips more legibly as JSON than as a 0-d object array.
ARRAY_KEYS = ("feature_names", "finding_ids", "group_ids", "X", "y", "y_optimistic",
              "impact", "observed")


def save_shard(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: payload[key] for key in ARRAY_KEYS}
    meta = {key: value for key, value in payload.items() if key not in ARRAY_KEYS}
    staged = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(staged, meta=np.array(json.dumps(meta)), **arrays)
    # Renamed last, so a shard file that exists is a shard file that is complete. An
    # interrupted write leaves a .partial.npz that the next run overwrites.
    staged.replace(path)


def load_shard(path: Path, *, flags: str) -> dict[str, Any] | None:
    """A previously built shard, or ``None`` if it is absent, damaged or out of date."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as handle:
            payload = {key: handle[key] for key in ARRAY_KEYS}
            payload.update(json.loads(str(handle["meta"])))
    except Exception as exc:  # a truncated or stale file is a rebuild, not a crash
        print(f"      {path.name} unreadable ({exc}); rebuilding", flush=True)
        return None
    if payload.get("format") != SHARD_FORMAT or payload.get("flags") != flags:
        print(f"      {path.name} was built by an older version; rebuilding", flush=True)
        return None
    return payload


def concatenate(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """One corpus from many shards, with group contiguity preserved.

    Scan ids are content hashes and shards are independently seeded, so no group spans two
    shards and plain concatenation keeps every scan's rows adjacent - which is what the
    ranking objective's positional grouping requires.
    """
    names = [str(name) for name in shards[0]["feature_names"]]
    for shard in shards[1:]:
        if [str(name) for name in shard["feature_names"]] != names:
            raise SystemExit("shards disagree on the feature columns; rebuild with --rebuild")
    out: dict[str, Any] = {
        key: np.concatenate([shard[key] for shard in shards]) for key in ARRAY_KEYS[1:]
    }
    out["feature_names"] = names
    for key in ("scans", "findings", "apps", "endpoints", "cves", "intel_covered",
                "circular_positives"):
        out[key] = int(sum(int(shard[key]) for shard in shards))
    return out


# ---------------------------------------------------------------------------
# Splitting and scoring
# ---------------------------------------------------------------------------


def time_ordered_split(group_ids: np.ndarray, observed: np.ndarray, fraction: float = 0.25):
    """Hold out the most recent scans, whole groups at a time.

    Splitting by row would put findings from one scan on both sides, which is not a
    generalisation test at all: the model would be asked to rank a scan it had partly seen.
    """
    latest: dict[str, float] = {}
    for group, when in zip(group_ids, observed):
        key = str(group)
        latest[key] = max(latest.get(key, float(when)), float(when))

    ordered = sorted(latest, key=lambda g: (latest[g], g))
    cut = int(len(ordered) * (1.0 - fraction))
    train_groups, test_groups = set(ordered[:cut]), set(ordered[cut:])
    train_mask = np.array([str(g) in train_groups for g in group_ids])
    return train_mask, ~train_mask, latest, ordered[:cut], ordered[cut:]


def group_sizes_for(group_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Contiguous group sizes for the selected rows, in row order."""
    sizes: list[int] = []
    current, count = None, 0
    for keep, group in zip(mask, group_ids):
        if not keep:
            continue
        if group != current:
            if count:
                sizes.append(count)
            current, count = group, 1
        else:
            count += 1
    if count:
        sizes.append(count)
    return np.asarray(sizes, dtype=int)


def frame_for(corpus: dict[str, Any], mask: np.ndarray, flags) -> FeatureFrame:
    matrix = pd.DataFrame(corpus["X"][mask], columns=corpus["feature_names"]).astype(float)
    return FeatureFrame(
        X=matrix,
        finding_ids=[str(f) for f in corpus["finding_ids"][mask]],
        group_ids=[str(g) for g in corpus["group_ids"][mask]],
        flags=flags,
    )


def as_date(seconds: float) -> str:
    return str(datetime.fromtimestamp(seconds, tz=timezone.utc).date())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=int, default=10_000)
    parser.add_argument("--shard-scans", type=int, default=500,
                        help="Scans per shard. Lower it when memory is tight.")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--min-findings", type=int, default=30)
    parser.add_argument("--max-findings", type=int, default=90)
    parser.add_argument("--out", type=Path, default=PROJECT / "data" / "synthetic-10k")
    parser.add_argument("--report", type=Path, default=PROJECT / "data" / "corpus-audit.json")
    parser.add_argument("--rebuild", action="store_true",
                        help="Ignore shards already on disk and build every one again.")
    parser.add_argument("--no-save-model", action="store_true",
                        help="Audit only; do not write the fitted ranker into the package.")
    args = parser.parse_args()

    config = load_config(PROJECT / "configs" / "offline.yaml")
    flags = config.flags()
    per_shard = max(1, min(args.shard_scans, args.scans))
    n_shards = max(1, math.ceil(args.scans / per_shard))
    shard_dir = args.out / "shards"

    print(f"corpus: {args.scans} scans in {n_shards} shard(s) of up to {per_shard}", flush=True)
    shards: list[dict[str, Any]] = []
    for index in range(n_shards):
        path = shard_dir / f"shard_{index:04d}.npz"
        existing = None if args.rebuild else load_shard(path, flags=flags.label())
        if existing is not None:
            print(f"  [{index + 1}/{n_shards}] reusing {path.name} "
                  f"({len(existing['y'])} rows)", flush=True)
            shards.append(existing)
            continue

        wanted = min(per_shard, args.scans - index * per_shard)
        print(f"  [{index + 1}/{n_shards}] building {wanted} scans ...", flush=True)
        payload = build_shard(
            wanted,
            (args.min_findings, args.max_findings),
            args.seed + index * 10_007,
            args.out / "work",
        )
        save_shard(path, payload)
        print(f"      {payload['findings']} findings, "
              f"{payload['intel_covered']} with intelligence, "
              f"{payload['seconds']:.0f}s -> {path.name}", flush=True)
        shards.append(payload)

    corpus = concatenate(shards)
    labels = corpus["y"].astype(float)
    optimistic = corpus["y_optimistic"].astype(float)
    group_ids = corpus["group_ids"]
    values = corpus["X"].astype(float)
    groups = group_sizes_for(group_ids, np.ones(len(group_ids), dtype=bool))

    report: dict[str, Any] = {"scale": {
        "scans": corpus["scans"],
        "findings": corpus["findings"],
        "applications": corpus["apps"],
        "endpoints": corpus["endpoints"],
        "cves": corpus["cves"],
        "rows": len(labels),
        "features": len(corpus["feature_names"]),
        "query_groups": len(groups),
        "shards": len(shards),
    }}
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)

    print("\naudit", flush=True)

    check("at least 10,000 scans",
          corpus["scans"] >= 10_000,
          f"{corpus['scans']} scans, {corpus['findings']} findings")

    rate = float((labels > 0).mean())
    n_independent = int((labels > 0).sum())
    n_circular = int(corpus["circular_positives"])
    report["positive_rate"] = rate
    report["optimistic_positive_rate"] = float((optimistic > 0).mean())
    report["independent_positives"] = n_independent
    report["circular_positives"] = n_circular
    check("positives are rare but present",
          MIN_POSITIVE_RATE <= rate <= MAX_POSITIVE_RATE,
          f"{rate:.3%} of findings were independently confirmed exploited "
          f"(want {MIN_POSITIVE_RATE:.0%}-{MAX_POSITIVE_RATE:.0%})")

    check("enough positives that no feature column already contains",
          n_independent >= MIN_INDEPENDENT_POSITIVES,
          f"{n_independent} oracle-backed positives, {n_circular} more resting only on KEV "
          f"or exploit evidence and therefore not scored "
          f"(want at least {MIN_INDEPENDENT_POSITIVES})")

    sizes = np.asarray(groups, dtype=float)
    report["group_sizes"] = {
        "mean": float(sizes.mean()), "min": int(sizes.min()), "max": int(sizes.max()),
        "p10": float(np.percentile(sizes, 10)), "p90": float(np.percentile(sizes, 90)),
    }
    check("query groups are big enough to rank within",
          sizes.mean() >= MIN_MEAN_GROUP_SIZE,
          f"mean {sizes.mean():.1f} findings per scan (min {int(sizes.min())}, "
          f"max {int(sizes.max())})")

    variances = values.var(axis=0)
    dead = [n for n, v in zip(corpus["feature_names"], variances) if v < MIN_FEATURE_VARIANCE]
    report["dead_features"] = dead
    check("every feature carries signal",
          not dead,
          f"all {len(corpus['feature_names'])} features vary" if not dead
          else f"{len(dead)} constant: {dead[:6]}")

    finite = bool(np.isfinite(values).all())
    check("no NaN or infinity in the matrix", finite,
          "all values finite" if finite else "non-finite values present")

    # The leakage check. A single observable that separates the label almost perfectly means
    # the corpus is teaching a lookup, not a ranking.
    aucs = sorted(
        ((auc(values[:, i], labels), n) for i, n in enumerate(corpus["feature_names"])),
        reverse=True,
    )
    report["top_single_feature_auc"] = [{"feature": n, "auc": round(a, 4)} for a, n in aucs[:8]]
    best_auc, best_name = aucs[0]
    check("no single feature gives the answer away",
          best_auc <= MAX_SINGLE_FEATURE_AUC,
          f"strongest is {best_name} at AUC {best_auc:.3f} "
          f"(ceiling {MAX_SINGLE_FEATURE_AUC})")

    # Generalisation: train on the past, rank the future, whole scans at a time.
    train_mask, test_mask, latest, train_groups, test_groups = time_ordered_split(
        group_ids, corpus["observed"]
    )
    train_last = max(latest[g] for g in train_groups)
    test_first = min(latest[g] for g in test_groups)
    report["split"] = {
        "train_groups": len(train_groups), "test_groups": len(test_groups),
        "train_last": as_date(train_last), "test_first": as_date(test_first),
    }
    check("the split is time-ordered with no overlap", train_last <= test_first,
          f"train ends {as_date(train_last)}, test starts {as_date(test_first)}")

    print("\ntraining on the past, ranking the future ...", flush=True)
    started = time.time()
    train_frame = frame_for(corpus, train_mask, flags)
    test_frame = frame_for(corpus, test_mask, flags)

    # Fitted on the training scans alone. Fitting on everything and then reporting the
    # metric over a slice of it is not a held-out number, and the previous version of this
    # script did exactly that.
    #
    # Fitted on the honest labels, for the same reason they are the ones reported: a model
    # taught that "in KEV" means "exploited" learns to restate its own b_kev column, and
    # graded against outcomes that column does not contain, that lesson is worth less than
    # nothing.
    model = build_ranker(config, config.ranking.ranker, int(config.ranking.seed))
    weights = None
    if config.ranking.impact_weighted_pairs:
        weights = 1.0 + np.log1p(np.maximum(0.0, corpus["impact"][train_mask]) / 1000.0)
    model.fit(train_frame, labels[train_mask], weights, int(config.ranking.seed))
    if getattr(model, "used_fallback", False):
        raise SystemExit("the ranker did not fit: "
                         + "; ".join(getattr(model, "warnings", ()) or ("no reason recorded",)))
    print(f"  fitted on {int(train_mask.sum())} rows in {time.time() - started:.0f}s", flush=True)

    test_sizes = group_sizes_for(group_ids, test_mask)
    scores = model.score(test_frame)
    held_out = mean_ndcg(scores, labels[test_mask], test_sizes, k=10)

    baselines: dict[str, float] = {}
    for column in ("b_epss", "b_kev", "cvss_base_max", "b_expected_loss_log",
                   "c_reach_delta_log", "scanner_severity_ord"):
        if column in test_frame.X.columns:
            series = test_frame.X[column].to_numpy(dtype=float)
            baselines[column] = mean_ndcg(series, labels[test_mask], test_sizes, k=10)
    rng = np.random.default_rng(args.seed)
    baselines["random"] = mean_ndcg(rng.random(int(test_mask.sum())), labels[test_mask],
                                    test_sizes, k=10)

    best_baseline = max(baselines.items(), key=lambda kv: kv[1])
    # The same predictions, graded against the leaky label set. Published for contrast: it
    # is the number this script used to print, and the distance between the two is how much
    # of that number was the model reading back its own input.
    inflated = mean_ndcg(scores, optimistic[test_mask], test_sizes, k=10)
    report["held_out_ndcg_at_10"] = {"lambdamart": held_out, **baselines}
    report["circularity"] = {
        "ndcg_at_10_against_leaky_labels": inflated,
        "delta": inflated - held_out,
        "note": ("The headline metric excludes positives justified only by KEV or exploit "
                 "evidence, both of which the ranker reads as features. This line is the "
                 "same model scored with those positives included, and is not a result. A "
                 "model trained on the honest labels should score worse here, because it "
                 "was not taught to restate its own b_kev column."),
    }
    print(f"  headline NDCG@10 {held_out:.3f}; against the leaky label set the same model "
          f"reads {inflated:.3f} ({inflated - held_out:+.3f})", flush=True)

    check("the learned ranker beats the best single signal",
          held_out - best_baseline[1] >= MIN_LIFT_OVER_BEST_BASELINE,
          f"lambdamart {held_out:.3f} vs {best_baseline[0]} {best_baseline[1]:.3f} "
          f"(+{held_out - best_baseline[1]:.3f}, want +{MIN_LIFT_OVER_BEST_BASELINE})")

    report["checks"] = checks
    report["passed"] = all(c["pass"] for c in checks)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\naudit written to {args.report}", flush=True)

    # A model is only shipped when every check passed. Shipping one from a corpus that
    # failed its own audit is how an unusable number ends up in a paper.
    if report["passed"] and not args.no_save_model:
        destination = Path(config.ranking.model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        model.save(destination)
        print(f"model written to {destination}", flush=True)

    print("RESULT:", "usable" if report["passed"] else "NOT usable - see the failures above")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
