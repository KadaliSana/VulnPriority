"""``vulnpriority``: the command line application (DESIGN.md 3.11).

The sixteen commands of DESIGN.md 3.11 - one per step of the framework, plus ``run-all`` and
the two bookkeeping commands - and the application-layer commands of DESIGN.md 6, which
register themselves onto the same application. Three conventions hold across all of them:

* ``--config`` names a YAML file (``configs/default.yaml`` when omitted, which is offline,
  deterministic and needs no API key), and ``--set key.path=value`` overrides any field of it
  from the command line, so an experiment does not require editing a checked-in file.
* ``--output-dir`` is the *runs root*; a run lives in ``<output-dir>/<run id>`` where the run
  id is derived from the configuration hash *and* a digest of the input the run was pointed
  at. The same configuration over the same scan therefore always resumes the same directory,
  while a changed configuration - or the same configuration over a different report - lands
  somewhere else instead of silently overwriting the evidence behind a published number, or
  reloading it and presenting it as this run's answer.
* A stage command runs everything it depends on, reloading whatever a previous invocation
  already completed. ``vulnpriority rank`` after ``vulnpriority enrich`` re-uses the enrichment
  rather than recomputing it.

Every import of a pipeline module happens inside a command body. That keeps ``--help``
instant, keeps the application usable while individual packages are still being built, and
keeps the offline guarantee honest: nothing is loaded until a command actually needs it.
"""

import codecs
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from vulnpriority.core.config import PROJECT_ROOT, PipelineConfig, load_config
from vulnpriority.core.money import (
    ASCII_FALLBACK_CHARS,
    DEFAULT_CURRENCY,
    currency_symbol,
    format_money,
    format_money_compact,
)

__all__ = ["app", "main"]

app = typer.Typer(
    name="vulnpriority",
    help=(
        "AI-driven web application vulnerability prioritization. "
        "Offline and deterministic by default: no network, no API key."
    ),
    add_completion=False,
    no_args_is_help=True,
)

DEFAULT_CONFIG = Path("configs/default.yaml")


@app.callback()
def _before_any_command() -> None:
    """Runs before every command, whichever entry point invoked the application.

    The only thing it does is make the output streams unable to abort a command on a
    character they cannot encode -- see :func:`_harden_streams`. It lives here rather than
    in :func:`main` because the installed console script is wired to ``app`` directly.
    """
    _harden_streams()


# ---------------------------------------------------------------------------
# Shared option plumbing
# ---------------------------------------------------------------------------


def _parse_overrides(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """``--set a.b=1`` pairs into the dotted-key mapping ``load_config`` applies.

    Values are parsed as JSON when possible so ``--set ranking.n_estimators=500`` yields an
    integer and ``--set component_c.enabled=false`` yields a boolean, rather than strings
    that a strict pydantic model would reject.
    """
    overrides: Dict[str, Any] = {}
    for pair in pairs or ():
        key, separator, raw = str(pair).partition("=")
        if not separator:
            raise typer.BadParameter(f"--set expects key=value, got {pair!r}")
        try:
            overrides[key.strip()] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            overrides[key.strip()] = raw
    return overrides


def _config(
    config: Optional[Path],
    overrides: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> PipelineConfig:
    """Load the configuration, apply ``--set`` overrides and honour ``--output-dir``."""
    path = config if config is not None else DEFAULT_CONFIG
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    loaded = (
        load_config(resolved, _parse_overrides(overrides))
        if resolved.exists()
        else PipelineConfig.model_validate(_nested(_parse_overrides(overrides)))
    )
    if output_dir is not None:
        loaded = loaded.model_copy(update={"output_dir": Path(output_dir)})
    return loaded


def _nested(flat: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dotted, value in flat.items():
        cursor = out
        *keys, last = dotted.split(".")
        for key in keys:
            cursor = cursor.setdefault(key, {})
        cursor[last] = value
    return out


def _with_dataset(config: PipelineConfig, dataset_dir: Optional[Path]):
    """Point the feeds at a synthetic dataset's fixtures and return its oracle-bearing object."""
    if dataset_dir is None:
        return config, None
    from vulnpriority.synth.generator import SyntheticDataset

    dataset = SyntheticDataset.load(dataset_dir)
    feeds = config.feeds.model_copy(update={"fixture_dir": Path(dataset.fixture_dir)})
    return config.model_copy(update={"feeds": feeds}), dataset


# ---------------------------------------------------------------------------
# The console's encoding
#
# A Windows console still hands Python a cp1252 stream by default, and U+20B9 (the rupee
# sign) is not in cp1252. Writing one to such a stream raises UnicodeEncodeError from
# inside ``typer.echo``, which killed ``run-all`` at the budget table *after* the pipeline
# had done all of its work - reporting a successful run as a crash.
#
# Two defences, because they fail differently. ``_harden_streams`` makes it impossible for
# any character to raise, including the ones that arrive embedded in data this module did
# not format (a reason code, an impact rationale). ``_ascii_console`` decides whether this
# module should write the symbol at all, which is what stops those amounts degrading to
# "?" in the common case.
# ---------------------------------------------------------------------------


def _encodable(text: str, stream: Any) -> bool:
    """Whether ``stream`` can carry ``text`` as it is currently configured."""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True          # a stream with no declared encoding (a test capture) is fine
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _ascii_console(currency: str) -> bool:
    """Whether this console needs the ASCII spelling of ``currency``'s symbol.

    Asked of the real symbol for the currency actually in force, so a USD run on a cp1252
    console still prints ``$``: only the symbol that cannot be encoded is replaced.

    Deliberately *not* consulted for the report or the web payload. Those are written as
    UTF-8 to a file and always carry the real glyph; this is a fact about the terminal in
    front of the operator, and it must not leak into an artefact that outlives it.
    """
    symbol = currency_symbol(currency)
    return not all(_encodable(symbol, stream) for stream in (sys.stdout, sys.stderr))


#: Name of the codec error handler registered by :func:`_harden_streams`.
_CURRENCY_ERRORS = "vulnpriority.currency"


def _spell_out_currency(error: UnicodeError) -> tuple[str, int]:
    """Codec error handler: write an unencodable currency symbol out in ASCII.

    Reached only for characters the stream genuinely cannot encode, and only for the
    embedded case -- an amount this module formatted itself never gets here, because
    :func:`_ascii_console` already chose a spelling the stream can carry. Anything that is
    not a currency symbol still becomes "?", which is what ``errors="replace"`` would have
    done anyway.
    """
    if not isinstance(error, UnicodeEncodeError):  # pragma: no cover - encode path only
        raise error
    chunk = error.object[error.start:error.end]
    return "".join(ASCII_FALLBACK_CHARS.get(char, "?") for char in chunk), error.end


def _harden_streams() -> None:
    """Make it impossible for an unencodable character to abort a command.

    The encoding is left alone on purpose. Forcing UTF-8 onto a console still running a
    legacy code page does not make the glyph appear - it makes the console decode UTF-8
    bytes as cp1252 and print mojibake, which is a worse answer than a question mark.
    ``errors="replace"`` keeps every other character intact and costs one glyph in the rare
    case where a rupee sign is embedded in text this module did not format itself.

    Guarded throughout: ``reconfigure`` is absent on a wrapped or redirected stream, and a
    stream that will not be reconfigured is not a reason to refuse to run.
    """
    try:
        codecs.lookup_error(_CURRENCY_ERRORS)
    except LookupError:
        codecs.register_error(_CURRENCY_ERRORS, _spell_out_currency)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        for errors in (_CURRENCY_ERRORS, "replace"):
            try:
                reconfigure(errors=errors)
                break
            except (AttributeError, LookupError, OSError, ValueError):
                continue  # pragma: no cover - stream-dependent


def _money(value: Any, currency: str) -> str:
    """A grouped amount for a terminal table, in a spelling this console can print."""
    return format_money(value, currency, ascii_only=_ascii_console(currency))


def _money_phrase(value: Any, currency: str) -> str:
    """An amount for terminal prose: lakh and crore, in a spelling this console can print."""
    return format_money_compact(value, currency, ascii_only=_ascii_console(currency))


def _currency(config: PipelineConfig) -> str:
    """What this run's money figures are denominated in.

    ``ImpactModel.currency`` is the authority, and Component B's resolver is the only thing
    that knows whether an inline override or a named preset is in force, so this asks it
    rather than guessing. A preset that will not load falls back to the package default --
    the same default the numbers would have been computed with.
    """
    try:
        return config.currency()
    except Exception:
        return DEFAULT_CURRENCY


def _echo_table(title: str, rows: Dict[str, Any]) -> None:
    typer.echo(title)
    width = max((len(str(key)) for key in rows), default=0)
    for key, value in rows.items():
        typer.echo(f"  {str(key).ljust(width)}  {value}")


def _run_through(
    stage: str,
    config: PipelineConfig,
    dataset: Any,
    *,
    scan_paths: Optional[List[Path]] = None,
    keep_going: bool = False,
):
    """Run ``stage`` and everything it depends on, resuming whatever is already done.

    Dependencies, not position: ``vulnpriority select`` needs a ranking and labels, and must not
    fail because the dataset was too small to cut evaluation folds from.
    """
    import sys

    from vulnpriority.pipeline.runner import PipelineRunner

    wanted = [stage]
    runner = PipelineRunner(
        config=config,
        dataset=dataset,
        command=" ".join(["vulnpriority"] + sys.argv[1:]),
        strict=not keep_going,
    )
    artifacts = runner.run(config, stages=wanted, scan_paths=scan_paths)
    if runner.errors:
        for name, message in runner.errors.items():
            typer.echo(f"  ! stage {name} unavailable: {message}", err=True)
    # Before the numbers, not after: a caveat that arrives last has already been read past.
    _echo_freshness(runner)
    return artifacts


def _fail(error: Exception) -> None:
    typer.echo(f"error: {error}", err=True)
    raise typer.Exit(code=1)


def _echo_ranking_provenance(ranking: Any) -> None:
    """Say how the ordering was produced, because the ranker's name does not.

    ``lambdamart`` is the policy that was asked for. Whether a model produced the order is
    a different question, and for a single application the answer used to be no while the
    artifact still said ``lambdamart`` - a learned ranking is a claim, and it was being made
    without grounds. One line, printed with the queue, settles it.
    """
    if ranking is None:
        return
    if getattr(ranking, "model_fitted", False):
        typer.echo("ranking: learned model, fitted on this run's own labelled data.")
    elif getattr(ranking, "model_pretrained", False):
        typer.echo(
            "ranking: learned model, trained on a labelled corpus and applied here. One scan "
            "is one query group, so nothing can be fitted from it; scoring needs no labels."
        )
    elif getattr(ranking, "fallback_reason", ""):
        typer.echo(
            f"ranking: NOT a learned model - {ranking.fallback_reason}, and no trained model "
            "was available. The queue is ordered by expected loss. Run `vulnpriority "
            "train-ranker --synthetic` to fit one."
        )


def _echo_budget_table(selections: Any, currency: str = DEFAULT_CURRENCY) -> None:
    """The Gap 10 headline: what each policy captures for the same budget.

    A single row ("57% of risk captured in 38 hours") is not a result - it becomes one only
    beside what sorting by CVSS captures for the same hours on the same items. This is the
    number a reader should not have to open a JSON file to find.
    """
    from vulnpriority.pipeline.stages import selection_table

    rows = selection_table(list(selections or ()))
    if not rows:
        return
    typer.echo(
        "\nbudget-constrained selection: same items, same cost model, same budget per scan"
    )
    # The money column is measured, not guessed. Its width depends on the currency -- a
    # rupee total runs several digits longer than the dollar one it replaced -- and on
    # whether this console made us spell the symbol out. A constant would be wrong for some
    # run, and it would shear the column carrying the headline number of Gap 10.
    captured = [_money(row["risk_captured"], currency) for row in rows]
    money_width = max(len("risk captured"), max(len(cell) for cell in captured))
    typer.echo(
        f"  {'policy':<18}  {'method':<12}  {'items':>5}  {'hours':>6}  {'budget':>6}  "
        f"{'risk captured':>{money_width}}  {'share':>8}   exploited caught"
    )
    for row, money in zip(rows, captured):
        method = "/".join(row["method"])
        caught = f"{row['exploited_captured']}/{row['exploited_total']}"
        typer.echo(
            f"  {row['policy'].value:<18}  {method:<12}  {row['items']:>5}  "
            f"{row['hours']:>6.1f}  {row['budget_used']:>6.0%}  "
            f"{money:>{money_width}}  "
            f"{row['risk_capture_share']:>8.1%}   "
            f"{caught:>9} ({row['exploited_share']:.0%})"
        )
    prefixed_rows = [row for row in rows if not row["optimised"]]
    optimised = [row for row in rows if row["optimised"]]
    if optimised and prefixed_rows:
        typer.echo(
            "  rank_prefix walks the queue until the budget is spent, which is what a team "
            "does with a\n  ranked list; the optimiser needs a value and a cost per finding. "
            "The same ranker appears\n  under both, so ordering quality and cost-awareness "
            "can be read apart."
        )


def _echo_freshness(runner: Any) -> None:
    """Say which stages were recomputed and which came back from the run directory.

    The run id is the configuration hash, so a *configuration* change lands in a different
    directory and can never resume stale numbers. A *code* change cannot: the same config
    resumes the same artifacts, and a fixed stage will happily hand back the answer it gave
    before the fix. That has already cost two people a debugging session, so the run says
    out loud when a number is not fresh.
    """
    reused = list(getattr(runner, "reused", ()) or ())
    computed = list(getattr(runner, "computed", ()) or ())
    if not reused:
        return
    typer.echo(
        f"\n  computed this run: {', '.join(computed) if computed else 'nothing'}"
        f"\n  reused from {', '.join(reused)}: these numbers are from a previous run of the"
        f"\n  same configuration. Re-run with --no-resume after changing code."
    )


def _echo_exposure_table(
    simulations: Any, reference: Any = None, capacity: Optional[Dict[str, Any]] = None
) -> None:
    """Per-policy exposure, reporting the quantity an ordering actually controls.

    ``exposure_days_total`` is bounded below by remediation capacity: the work the budget
    cannot reach accrues the full horizon whatever order it is done in, so the total is
    near-identical under every policy and a comparison drawn on it reports noise. The three
    columns here are the ones an ordering moves - exposure carried by the findings that were
    really exploited, exploitations averted, and the reduction against the reference policy.
    Total exposure is still shown, labelled so nobody quotes it as a comparison.
    """
    rows = sorted(
        simulations or (),
        key=lambda item: (item.exposure_days_exploited, item.policy.value),
    )
    if not rows:
        return
    label = getattr(reference, "value", reference) or "reference"
    typer.echo("\nlongitudinal exposure (26 weeks, one policy per row)")
    typer.echo(
        f"  {'policy':<18}{'exploited exposure':>20}{'prevented':>12}"
        f"{'vs ' + str(label):>14}{'total (capacity-bound)':>24}"
    )
    for item in rows:
        prevented = f"{item.exploited_remediated_before_exploit}/{item.exploited_total}"
        reduction = (
            f"{item.reduction_vs_cvss:+.1%}" if item.reduction_vs_cvss is not None else "-"
        )
        typer.echo(
            f"  {item.policy.value:<18}{item.exposure_days_exploited:>18,.0f} d"
            f"{prevented:>12}{reduction:>14}{item.exposure_days_total:>22,.0f} d"
        )
    if capacity:
        available = float(capacity.get("weeks", 0)) * float(
            capacity.get("capacity_hours_per_week", 0.0)
        )
        backlog = float(capacity.get("backlog_hours", 0.0))
        reach = (min(1.0, available / backlog) if backlog > 0 else 1.0)
        typer.echo(
            f"  backlog {backlog:,.0f} h against {available:,.0f} h of capacity: the budget "
            f"can reach {reach:.1%} of the work,\n  so 'prevented' is out of what was "
            f"reachable at all, not out of what was ideal."
        )


# ---------------------------------------------------------------------------
# synth
# ---------------------------------------------------------------------------


@app.command("synth")
def synth(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Parent directory for the dataset (default data/synthetic)."
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override the synthetic seed."),
    apps: Optional[int] = typer.Option(None, "--apps", help="Number of applications to generate."),
    scans_per_app: Optional[int] = typer.Option(
        None, "--scans-per-app", help="Repeated scans per application, for time-ordered splits."
    ),
    injection_fraction: Optional[float] = typer.Option(
        None, "--injection-fraction", help="Fraction of reference pages carrying an injection payload."
    ),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Generate the synthetic world: scans, feed fixtures, reference pages, exploitation oracle."""
    from vulnpriority.synth.generator import DEFAULT_SYNTHETIC_ROOT, SyntheticDataset

    settings = _config(config, set_)
    synthetic = settings.synthetic
    updates: Dict[str, Any] = {}
    if seed is not None:
        updates["seed"] = int(seed)
    if apps is not None:
        updates["n_apps"] = int(apps)
    if scans_per_app is not None:
        updates["scans_per_app"] = int(scans_per_app)
    if injection_fraction is not None:
        updates["injection_fraction"] = float(injection_fraction)
    if updates:
        synthetic = synthetic.model_copy(update=updates)

    target = Path(output_dir) if output_dir is not None else DEFAULT_SYNTHETIC_ROOT
    try:
        dataset = SyntheticDataset.generate(synthetic, target)
    except Exception as error:  # noqa: BLE001 - surfaced to the user, not swallowed
        _fail(error)
        return
    _echo_table("synthetic dataset", dataset.summary())
    typer.echo(f"\nfeed fixtures: {dataset.fixture_dir}")
    typer.echo(f"use with:      vulnpriority run-all --dataset {dataset.root}")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@app.command("ingest")
def ingest(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    input_: Optional[List[Path]] = typer.Option(
        None, "--input", "-i", help="Scanner report file or directory (repeatable)."
    ),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Parse ZAP, Burp, Nuclei or canonical JSON into correlated Scan documents."""
    settings, world = _with_dataset(_config(config, set_, output_dir), dataset)
    try:
        artifacts = _run_through("ingest", settings, world, scan_paths=list(input_ or ()))
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    _echo_table(
        "ingest",
        {
            "run": artifacts.run_id,
            "directory": artifacts.root,
            "scans": len(artifacts.scans),
            "endpoints": sum(len(scan.endpoints) for scan in artifacts.scans),
            "findings": sum(len(scan.findings) for scan in artifacts.scans),
            "applications": len({scan.app_id for scan in artifacts.scans}),
        },
    )


# ---------------------------------------------------------------------------
# fetch-feeds
# ---------------------------------------------------------------------------


@app.command("fetch-feeds")
def fetch_feeds(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    as_of: Optional[str] = typer.Option(None, "--as-of", help="ISO date; nothing later is returned."),
    cve: Optional[List[str]] = typer.Option(None, "--cve", help="CVE id to fetch (repeatable)."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Assemble intelligence for a set of CVEs as of a date, warming the cache in live modes."""
    from datetime import date as date_type

    from vulnpriority.feeds.bundle import DefaultIntelAssembler, build_feed_bundle

    settings, world = _with_dataset(_config(config, set_, output_dir), dataset)
    cutoff = date_type.fromisoformat(as_of) if as_of else date_type.today()
    wanted = [item.strip().upper() for item in (cve or ()) if item.strip()]
    if not wanted and world is not None:
        wanted = [vuln.cve_id for vuln in world.world.vulns]
    if not wanted:
        _fail(ValueError("nothing to fetch: pass --cve or --dataset"))
        return
    try:
        assembler = DefaultIntelAssembler(build_feed_bundle(settings), settings)
        rows = []
        for cve_id in wanted:
            intel = assembler.assemble(cve_id, cutoff, settings.feeds.max_references_per_cve)
            rows.append(
                (
                    cve_id,
                    f"{max((record.base_score for record in intel.cvss), default=0.0):.1f}",
                    f"{intel.epss.score:.4f}" if intel.epss else "-",
                    "yes" if intel.kev and intel.kev.in_kev else "no",
                    str(len(intel.exploits)),
                    str(len(intel.references)),
                )
            )
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    typer.echo(f"feed mode {settings.feeds.mode.value}, as of {cutoff.isoformat()}")
    typer.echo(f"{'CVE':<20}{'CVSS':>6}{'EPSS':>9}{'KEV':>5}{'EXPL':>6}{'REFS':>6}")
    for row in rows[:50]:
        typer.echo(f"{row[0]:<20}{row[1]:>6}{row[2]:>9}{row[3]:>5}{row[4]:>6}{row[5]:>6}")
    if len(rows) > 50:
        typer.echo(f"... and {len(rows) - 50} more")


# ---------------------------------------------------------------------------
# assess / enrich / chain
# ---------------------------------------------------------------------------


@app.command("assess")
def assess(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    input_: Optional[List[Path]] = typer.Option(None, "--input", "-i", help="Scanner report (repeatable)."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Component A: asset criticality, exploitability and applicability, inside the sandbox."""
    settings, world = _with_dataset(_config(config, set_, output_dir), dataset)
    try:
        artifacts = _run_through("enrich", settings, world, scan_paths=list(input_ or ()))
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    admin = sum(1 for item in artifacts.enriched if item.asset.is_admin_surface)
    signals = sum(item.trust.injection_signal_count for item in artifacts.enriched)
    _echo_table(
        "assess (component A)",
        {
            "run": artifacts.run_id,
            "backend": settings.llm.backend.value,
            "component_a": "on" if settings.component_a.enabled else "off (structural only)",
            "findings assessed": len(artifacts.enriched),
            "admin surfaces": admin,
            "injection signals": signals,
        },
    )


@app.command("enrich")
def enrich(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    attacker: Optional[str] = typer.Option(None, "--attacker", help="Attacker preset name."),
    impact: Optional[str] = typer.Option(None, "--impact", help="Impact model preset name."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Component B: P(exploit) for this attacker, monetary impact, cost, expected loss."""
    overrides = list(set_ or ())
    if attacker:
        overrides.append(f"component_b.attacker_preset={attacker}")
    if impact:
        overrides.append(f"component_b.impact_preset={impact}")
    settings, world = _with_dataset(_config(config, overrides, output_dir), dataset)
    try:
        artifacts = _run_through("enrich", settings, world)
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    currency = _currency(settings)
    losses = sorted(artifacts.enriched, key=lambda item: -item.expected_loss)[:5]
    _echo_table(
        "enrich (component B)",
        {
            "run": artifacts.run_id,
            "attacker": settings.component_b.attacker_preset,
            "impact model": settings.component_b.impact_preset,
            "component_b": "on" if settings.component_b.enabled else "off (neutral priors)",
            "findings": len(artifacts.enriched),
            "total expected loss": _money_phrase(
                sum(item.expected_loss for item in artifacts.enriched), currency
            ),
        },
    )
    if losses:
        typer.echo("\ntop expected loss")
        for item in losses:
            typer.echo(
                f"  {item.finding_id}  p={item.likelihood.p_exploit:.3f}  "
                f"{_money(item.expected_loss, currency):>16}  {item.finding.name}"
            )


@app.command("chain")
def chain(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Component C: build the attack graph and score each finding's reachability contribution."""
    settings, world = _with_dataset(_config(config, set_, output_dir), dataset)
    try:
        artifacts = _run_through("chain", settings, world)
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    currency = _currency(settings)
    total_risk = sum(graph.total_risk for graph in artifacts.graphs)
    rejected = sum(graph.rejected_untrusted_edges for graph in artifacts.graphs)
    _echo_table(
        "chain (component C)",
        {
            "run": artifacts.run_id,
            "component_c": "on" if settings.component_c.enabled else "off (no chain features)",
            "graphs": len(artifacts.graphs),
            "nodes": sum(len(graph.nodes) for graph in artifacts.graphs),
            "edges": sum(len(graph.edges) for graph in artifacts.graphs),
            "rejected untrusted edges": rejected,
            "total reachable risk": _money_phrase(total_risk, currency),
        },
    )


# ---------------------------------------------------------------------------
# rank / explain
# ---------------------------------------------------------------------------


@app.command("train-ranker")
def train_ranker(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Generate a synthetic world first and train on it."
    ),
    dataset: Optional[Path] = typer.Option(
        None, "--dataset", help="Existing synthetic dataset directory to train on."
    ),
    scan: Optional[List[Path]] = typer.Option(
        None, "--scan", "-i", help="Scanner reports to train on (repeatable; needs several)."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Where to write the model (default: ranking.model_path)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Runs root directory for the stages this runs."
    ),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Fit the LambdaMART ranker on a labelled corpus and save it for later runs.

    Ranking a single application cannot fit a model: one scan is one query group, and a
    pairwise objective has no pair to learn from inside one group. So the model is fitted
    here, once, on a corpus that has several scans and confirmed-exploitation labels, and
    every later run scores with it. Without a saved model those runs order by expected loss
    and say so; with one, the ordering is the learned model's.
    """
    from vulnpriority.core.config import DEFAULT_RANKER_MODEL
    from vulnpriority.pipeline.stages import train_ranker_stage

    settings, world = _with_dataset(_config(config, set_, output_dir), dataset)
    if synthetic and world is None:
        from vulnpriority.synth.generator import DEFAULT_SYNTHETIC_ROOT, SyntheticDataset

        typer.echo("generating a synthetic world to train on...")
        try:
            generated = SyntheticDataset.generate(settings.synthetic, DEFAULT_SYNTHETIC_ROOT)
        except Exception as error:  # noqa: BLE001
            _fail(error)
            return
        # Re-resolve through the same helper the other commands use, so the feeds point at
        # this dataset's fixtures rather than at whatever the config named.
        settings, world = _with_dataset(settings, generated.root)

    try:
        artifacts = _run_through("label", settings, world, scan_paths=list(scan or ()))
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return

    scans = list(artifacts.scans or ())
    typer.echo(f"training on {len(scans)} scan(s), {len(artifacts.enriched)} finding(s)")
    try:
        model, frame = train_ranker_stage(
            settings, artifacts.enriched, artifacts.labels, artifacts.chain_scores
        )
    except Exception as error:  # noqa: BLE001 - a corpus that cannot teach is a user error
        _fail(error)
        return

    destination = Path(output) if output is not None else (
        settings.ranking.model_path or DEFAULT_RANKER_MODEL
    )
    model.save(destination)
    _echo_table("trained ranker", {
        "scans": len(scans),
        "findings": len(frame.finding_ids),
        "features": len(frame.feature_names),
        "query groups": len(frame.group_sizes()),
        "fitted": "yes" if not model.used_fallback else "no",
        "written to": destination,
    })
    typer.echo(
        "\nlater runs over a single scan will now score with this model rather than "
        "falling back to expected-loss ordering."
    )


@app.command("rank")
def rank(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    ranker: Optional[str] = typer.Option(None, "--ranker", help="Ranker name (default lambdamart)."),
    attacker: Optional[str] = typer.Option(None, "--attacker", help="Attacker preset name."),
    top: int = typer.Option(10, "--top", help="How many ranked findings to print."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Build features, run the ranker and print the explained remediation queue."""
    overrides = list(set_ or ())
    if ranker:
        overrides.append(f"ranking.ranker={ranker}")
    if attacker:
        overrides.append(f"component_b.attacker_preset={attacker}")
    settings, world = _with_dataset(_config(config, overrides, output_dir), dataset)
    try:
        artifacts = _run_through("rank", settings, world)
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    result = artifacts.ranking
    if result is None:
        _fail(RuntimeError("the rank stage produced no ranking"))
        return
    currency = _currency(settings)
    _echo_table(
        "rank",
        {
            "run": artifacts.run_id,
            "ranker": result.ranker.value,
            "cell": result.flags.label(),
            "features": 0 if artifacts.features is None else len(artifacts.features.X.columns),
            "ranked findings": len(result.items),
            "manipulation alerts": sum(1 for item in result.items if item.alerts),
        },
    )
    typer.echo("")
    for item in sorted(result.items, key=lambda row: row.rank)[: max(0, top)]:
        reasons = ", ".join(item.explanation.reason_codes[:3]) if item.explanation else ""
        flag = " [!]" if item.alerts else ""
        typer.echo(
            f"  {item.rank:>3}. {item.finding_id}  p={item.p_exploit:.3f}  "
            f"{_money(item.chain_adjusted_loss, currency):>16}{flag}  {reasons}"
        )


@app.command("explain")
def explain(
    finding: str = typer.Option(..., "--finding", "-f", help="Finding id to explain."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Show the factors behind one finding's position in the queue."""
    from vulnpriority.pipeline.artifacts import RunArtifacts, find_run_root

    settings, _world = _with_dataset(_config(config, set_, output_dir), dataset)
    root = find_run_root(settings)
    if not root.exists():
        _fail(FileNotFoundError(f"no run directory at {root}; run `vulnpriority rank` first"))
        return
    artifacts = RunArtifacts.load(root)
    ranked = next(
        (item for item in (artifacts.ranking.items if artifacts.ranking else ()) if item.finding_id == finding),
        None,
    )
    enriched = next((item for item in artifacts.enriched if item.finding_id == finding), None)
    currency = _currency(settings)
    if ranked is None and enriched is None:
        _fail(KeyError(f"{finding} is not in run {artifacts.run_id}"))
        return

    if enriched is not None:
        _echo_table(
            f"explain {finding}",
            {
                "name": enriched.finding.name,
                "endpoint": f"{enriched.endpoint.method.value} {enriched.endpoint.path}",
                "function": enriched.asset.function.value,
                "p(exploit)": f"{enriched.likelihood.p_exploit:.4f}",
                "impact": _money(enriched.impact.total, currency),
                "expected loss": _money(enriched.expected_loss, currency),
                "remediation": f"{enriched.remediation.hours:.1f} h",
                "max trust tier used": enriched.trust.max_tier_used.name,
                "injection signals": enriched.trust.injection_signal_count,
            },
        )
        typer.echo("\nlikelihood terms")
        for term, value in sorted(
            enriched.likelihood.log_odds_terms.items(), key=lambda pair: -abs(pair[1])
        ):
            typer.echo(f"  {term:<28} {value:+.4f}")

    explanation = ranked.explanation if ranked is not None else None
    if explanation is None:
        explanation = next(
            (item for item in artifacts.explanations if item.finding_id == finding), None
        )
    if ranked is not None:
        typer.echo(f"\nrank {ranked.rank} of {len(artifacts.ranking.items)}  score {ranked.score:.4f}")
    if explanation is not None:
        typer.echo("\ntop feature contributions")
        for contribution in explanation.top_contributions:
            typer.echo(
                f"  {contribution.feature:<28} value={contribution.value:+.4f} "
                f"shap={contribution.shap_value:+.4f} tier={contribution.tier.name}"
            )
        if explanation.reason_codes:
            typer.echo("\nreason codes")
            for code in explanation.reason_codes:
                typer.echo(f"  - {code}")
        typer.echo(
            f"\nuntrusted influence share: {explanation.untrusted_influence_share:.1%}"
        )


# ---------------------------------------------------------------------------
# evaluate / ablate
# ---------------------------------------------------------------------------


@app.command("evaluate")
def evaluate(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    split: Optional[str] = typer.Option(
        None, "--split", help="time_ordered (default), leave_one_app_out or random (control)."
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Evaluate every ranker on identical data against confirmed-exploitation labels."""
    overrides = list(set_ or ())
    if split:
        overrides.append(f"evaluation.split_kind={split}")
    settings, world = _with_dataset(_config(config, overrides, output_dir), dataset)
    try:
        artifacts = _run_through("evaluate", settings, world)
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    _echo_table(
        "evaluate",
        {
            "run": artifacts.run_id,
            "split": settings.evaluation.split_kind.value,
            "folds": settings.evaluation.n_folds,
            "gap days": settings.evaluation.gap_days,
            "labels": 0 if artifacts.labels is None else len(artifacts.labels.labels),
            "positives": 0 if artifacts.labels is None else len(artifacts.labels.positives()),
            "metric bundles": len(artifacts.metrics),
        },
    )
    for bundle in artifacts.metrics:
        values = bundle.as_dict()
        headline = {key: values[key] for key in sorted(values) if key.startswith(("ndcg", "pr_auc", "mcc"))}
        rendered = "  ".join(f"{key}={value:.4f}" for key, value in headline.items())
        typer.echo(f"  {bundle.ranker.value:<18} [{bundle.flags.label():<4}] {rendered}")


@app.command("ablate")
def ablate(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Full 2^3 factorial ablation over components A, B and C, with main effects."""
    from vulnpriority.pipeline.stages import ablate_stage

    settings, world = _with_dataset(_config(config, set_, output_dir), dataset)
    try:
        artifacts = _run_through("label", settings, world)
        table = ablate_stage(settings, artifacts.scans, artifacts.enriched, artifacts.labels)
        artifacts.ablation = table
        artifacts.save(artifacts.root)
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    typer.echo(f"ablation over {len(getattr(table, 'cells', ()))} cells -> {artifacts.root}")
    for cell in getattr(table, "cells", ()):
        rendered = "  ".join(f"{key}={value:.4f}" for key, value in sorted(cell.mean.items())[:4])
        typer.echo(f"  [{cell.flags.label():<4}] n={cell.n:<3} {rendered}")
    for effect, values in getattr(table, "main_effects", {}).items():
        rendered = "  ".join(f"{key}={value:+.4f}" for key, value in sorted(values.items())[:4])
        typer.echo(f"  main effect {effect}: {rendered}")


# ---------------------------------------------------------------------------
# select / simulate / adversarial
# ---------------------------------------------------------------------------


@app.command("select")
def select(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    budget_hours: Optional[float] = typer.Option(None, "--budget-hours", help="Remediation budget."),
    method: Optional[str] = typer.Option(
        None, "--method", help="dp_exact (default), greedy_ratio or rank_prefix."
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Knapsack selection under a remediation budget, charged once per root cause."""
    overrides = list(set_ or ())
    if budget_hours is not None:
        overrides.append(f"selection.budget_hours={budget_hours}")
    if method:
        overrides.append(f"selection.method={method}")
    settings, world = _with_dataset(_config(config, overrides, output_dir), dataset)
    try:
        artifacts = _run_through("select", settings, world)
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    scans = len({result.scan_id for result in artifacts.selection})
    _echo_table(
        "select",
        {
            "run": artifacts.run_id,
            "budget": f"{settings.selection.budget_hours:.1f} h per scan",
            "method": f"{settings.selection.method.value} (baselines: rank_prefix)",
            "scans": scans,
            "policies": len({result.ranker for result in artifacts.selection}),
        },
    )
    _echo_ranking_provenance(artifacts.ranking)
    _echo_budget_table(artifacts.selection, _currency(settings))


@app.command("simulate")
def simulate(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    weeks: Optional[int] = typer.Option(None, "--weeks", help="Simulation horizon in weeks."),
    capacity: Optional[float] = typer.Option(
        None, "--capacity", help="Remediation capacity in hours per week."
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Longitudinal exposure simulation per policy: exposure days, not prediction accuracy."""
    overrides = list(set_ or ())
    if weeks is not None:
        overrides.append(f"simulation.weeks={weeks}")
    if capacity is not None:
        overrides.append(f"simulation.capacity_hours_per_week={capacity}")
    settings, world = _with_dataset(_config(config, overrides, output_dir), dataset)
    try:
        artifacts = _run_through("simulate", settings, world)
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    _echo_table(
        "simulate",
        {
            "run": artifacts.run_id,
            "weeks": settings.simulation.weeks,
            "capacity": f"{settings.simulation.capacity_hours_per_week:.1f} h/week",
            "policies": len(artifacts.simulation),
        },
    )
    _echo_exposure_table(
        artifacts.simulation,
        settings.simulation.reference_policy,
        artifacts.simulation_capacity,
    )


@app.command("adversarial")
def adversarial(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    corpus: Optional[Path] = typer.Option(None, "--corpus", help="Adversarial corpus YAML."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Run the prompt-injection and rank-manipulation corpus against the guarded pipeline."""
    overrides = list(set_ or ())
    if corpus is not None:
        overrides.append(f"adversarial.corpus_path={corpus}")
    settings, world = _with_dataset(_config(config, overrides, output_dir), dataset)
    try:
        artifacts = _run_through("adversarial", settings, world)
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    report = artifacts.adversarial
    if report is None:
        _fail(RuntimeError("the adversarial stage produced no report"))
        return
    _echo_table(
        "adversarial",
        {
            "run": artifacts.run_id,
            "corpus": settings.adversarial.corpus_path,
            "cases": report.n_cases,
            "attack success rate": f"{report.attack_success_rate:.1%}",
            "canary leak rate": f"{report.canary_leak_rate:.1%}",
            "detection rate": f"{report.detection_rate:.1%}",
            "false positive rate": f"{report.false_positive_rate:.1%}",
            "max rank shift": report.max_abs_rank_shift,
        },
    )
    breached = (
        report.attack_success_rate > settings.adversarial.max_attack_success_rate
        or report.canary_leak_rate > settings.adversarial.max_canary_leak_rate
        or report.detection_rate < settings.adversarial.min_detection_rate
        or report.false_positive_rate > settings.adversarial.max_false_positive_rate
    )
    if breached:
        typer.echo("\nthresholds breached: the configured robustness budget was exceeded", err=True)
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# report / run-all / manifest
# ---------------------------------------------------------------------------


@app.command("report")
def report(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Synthetic dataset directory."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Write report.md, report.json and the figures for an existing run."""
    settings, world = _with_dataset(_config(config, set_, output_dir), dataset)
    try:
        artifacts = _run_through("report", settings, world)
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    typer.echo(f"report written to {artifacts.root}")
    for name in sorted(path.name for path in Path(artifacts.root).glob("report*")):
        typer.echo(f"  {name}")


def _write_assessment_report(artifacts: Any, runner: Any) -> None:
    """Write the operator's report for a run with no evaluation behind it.

    The research report needs metrics, baselines and an ablation. A single scan has none of
    those and still deserves a document: what was found, what to fix first, what the budget
    reaches, and what that leaves. This is that document.
    """
    try:
        from vulnpriority.report import build_report
        from vulnpriority.web.exporter import build_dashboard
    except ImportError as error:  # pragma: no cover - only when an extra is absent
        typer.echo(f"  (no assessment report: {error})")
        return

    try:
        data = build_dashboard(
            scans=artifacts.scans or (),
            enriched=artifacts.enriched or (),
            chain=(artifacts.chain_scores or {}),
            graphs=artifacts.graphs or (),
            ranking=artifacts.ranking,
            selections=artifacts.selection or (),
            manifest=artifacts.manifest,
            config=runner.config,
        )
        destination = Path(artifacts.root) / "assessment.md"
        destination.write_text(build_report(data, "md"), encoding="utf-8")
        html = Path(artifacts.root) / "assessment.html"
        html.write_text(build_report(data, "html"), encoding="utf-8")
    except Exception as error:  # noqa: BLE001 - a report failure must not lose the run
        typer.echo(f"  (assessment report not written: {error})")
        return
    typer.echo(f"\nassessment report: {destination}")
    typer.echo(f"                   {html}  (open in a browser, print to PDF)")


@app.command("run-all")
def run_all(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Generate a synthetic world first and run the whole chain on it."
    ),
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Existing synthetic dataset directory."),
    scan: Optional[List[Path]] = typer.Option(None, "--scan", "-i", help="Scanner report (repeatable)."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    keep_going: bool = typer.Option(
        False, "--keep-going", help="Report an unavailable stage and continue instead of failing."
    ),
    no_resume: bool = typer.Option(False, "--no-resume", help="Recompute every stage from scratch."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Run the whole chain: ingest, assess, enrich, chain, rank, evaluate, select, simulate, report."""
    import sys

    from vulnpriority.pipeline.runner import PipelineRunner
    from vulnpriority.synth.generator import SyntheticDataset

    settings = _config(config, set_, output_dir)
    world = None
    if synthetic and dataset is None:
        root = Path(settings.data_dir)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        world = SyntheticDataset.generate(settings.synthetic, root / "synthetic")
        typer.echo(f"generated synthetic dataset {world.dataset_hash} at {world.root}")
        feeds = settings.feeds.model_copy(update={"fixture_dir": Path(world.fixture_dir)})
        settings = settings.model_copy(update={"feeds": feeds})
    elif dataset is not None:
        settings, world = _with_dataset(settings, dataset)

    if world is None and not scan:
        _fail(ValueError("nothing to run on: pass --synthetic, --dataset or --scan"))
        return

    # Evaluation, ablation and the exposure simulation all need labelled history across
    # several scans: time-ordered splits cannot be cut from one. Assessing a single
    # application is the ordinary operational case, so those stages are dropped rather than
    # allowed to fail the run - an operator who scanned their own app should not be told
    # their scan is unusable because a research protocol could not be satisfied.
    stages = None
    if world is None and len(list(scan or ())) < settings.evaluation.min_train_scans:
        from vulnpriority.pipeline.runner import STAGE_ORDER

        # ``report`` goes too: that stage writes the RESEARCH report, whose subject is the
        # evaluation, so it depends on the stages being dropped. The operator's document is
        # the assessment report, written below from the ranking that did run.
        # ``adversarial`` too: it measures the FRAMEWORK's robustness against a fixed
        # corpus, not anything about the scan in hand, and its corpus needs vulnerability
        # intelligence to inject into. Run it deliberately with `vulnpriority adversarial`.
        research_only = {
            "label", "split", "evaluate", "ablate", "simulate", "adversarial", "report",
        }
        stages = [name for name in STAGE_ORDER if name not in research_only]
        typer.echo(
            f"single-scan run: ranking, budget selection and the report only.\n"
            f"  Evaluation and the exposure simulation need at least "
            f"{settings.evaluation.min_train_scans} scans of labelled history to compare against."
        )

    runner = PipelineRunner(
        config=settings,
        dataset=world,
        command=" ".join(["vulnpriority"] + sys.argv[1:]),
        strict=not keep_going,
    )
    try:
        artifacts = runner.run(
            settings, stages=stages, scan_paths=list(scan or ()), resume=not no_resume
        )
    except Exception as error:  # noqa: BLE001
        _fail(error)
        return
    _echo_table("run-all", artifacts.summary())
    _echo_freshness(runner)
    if stages is not None:
        _write_assessment_report(artifacts, runner)
    _echo_ranking_provenance(artifacts.ranking)
    _echo_budget_table(artifacts.selection, _currency(settings))
    _echo_exposure_table(
        artifacts.simulation,
        settings.simulation.reference_policy,
        artifacts.simulation_capacity,
    )
    if runner.errors:
        typer.echo("\nunavailable stages")
        for name, message in runner.errors.items():
            typer.echo(f"  {name}: {message}")


@app.command("manifest")
def manifest(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Pipeline config YAML."),
    dataset: Optional[Path] = typer.Option(
        None, "--dataset", help="Synthetic dataset directory, so the run id matches that run."
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Runs root directory."),
    run_dir: Optional[Path] = typer.Option(None, "--run-dir", help="Explicit run directory."),
    as_json: bool = typer.Option(False, "--json", help="Print the manifest as JSON."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="Override a config field: key.path=value."),
) -> None:
    """Print a run's manifest: config hash, seeds, dataset hash and library versions."""
    from vulnpriority.pipeline.artifacts import find_run_root, load_manifest

    settings, _world = _with_dataset(_config(config, set_, output_dir), dataset)
    root = Path(run_dir) if run_dir is not None else find_run_root(settings)
    try:
        record = load_manifest(root)
    except (FileNotFoundError, OSError) as error:
        _fail(FileNotFoundError(f"no manifest in {root}: {error}"))
        return
    if as_json:
        typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))
        return
    _echo_table(
        f"manifest {record.run_id}",
        {
            "created": record.created_at.isoformat(timespec="seconds"),
            "config hash": record.config_hash,
            "dataset hash": record.dataset_hash,
            "seeds": ", ".join(str(seed) for seed in record.seeds),
            "llm backend": f"{record.llm_backend.value} ({record.llm_model})",
            "feed mode": record.feed_mode.value,
            "as of": record.as_of.isoformat() if record.as_of else "-",
            "command": record.command,
        },
    )
    typer.echo("\npackage versions")
    for name, version in record.package_versions.items():
        typer.echo(f"  {name:<16} {version}")


@app.command("web")
def web(
    run_dir: Optional[Path] = typer.Option(None, "--run", "-r", help="Run directory to read artifacts from."),
    output: Path = typer.Option(Path("site"), "--output", "-o", help="Where to write the static site."),
    demo: bool = typer.Option(False, "--demo", help="Render the worked example instead of a real run."),
    serve: bool = typer.Option(False, "--serve", help="Serve the exported site and print its URL."),
    port: int = typer.Option(8000, "--port", help="Port for --serve."),
) -> None:
    """Export the results website: one page, one data file, opens from disk with no server."""
    from vulnpriority.web.exporter import export_site

    if demo:
        from vulnpriority.web.demo import demo_dashboard

        data = demo_dashboard()
    elif run_dir is not None:
        from vulnpriority.web.loader import load_run

        try:
            data = load_run(run_dir)
        except (FileNotFoundError, NotADirectoryError) as error:
            _fail(error)
            return
    else:
        _fail(ValueError("give --run <dir> to export a run, or --demo to render the worked example"))
        return

    site = export_site(data, output)
    typer.echo(f"site written to {site.resolve()}")
    typer.echo(f"  {len(data.findings)} findings, {len(data.graphs)} attack graph(s), {len(data.metrics)} metric rows")
    if serve:
        from vulnpriority.web.server import serve_site

        _httpd, url = serve_site(site, port=port, open_browser=True)
        typer.echo(f"serving at {url} (ctrl-c to stop)")


@app.command("serve")
def serve_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind. Localhost by default: this is a local tool."),
    port: int = typer.Option(8765, "--port", "-p", help="Port to listen on."),
    open_browser: bool = typer.Option(False, "--open", help="Open the site once the server is up."),
    no_scan: bool = typer.Option(False, "--no-scan", help="Refuse target assessment; uploads only."),
    allowed_host: Optional[List[str]] = typer.Option(
        None, "--allowed-host",
        help=(
            "Additional hostname to accept in the Host header (repeatable). Loopback names "
            "and the bind address are always accepted; anything else is refused, which is "
            "what stops a page on another site from reaching this server by pointing its "
            "own hostname at this machine."
        ),
    ),
) -> None:
    """Run the interactive site: upload a scanner report, or assess a target you own."""
    try:
        from vulnpriority.web.app import serve
    except ImportError as error:
        _fail(ImportError(f'the web extra is not installed: pip install -e ".[web]" ({error})'))
        return

    typer.echo(f"vulnpriority is at http://{host}:{port}  (ctrl-c to stop)")
    if host not in ("127.0.0.1", "localhost", "::1"):
        typer.echo("  bound beyond localhost: this server has no authentication, so put it behind one")
    serve(
        host=host, port=port, open_browser=open_browser, allow_scan=not no_scan,
        allowed_hosts=tuple(allowed_host or ()),
    )


@app.command("scan")
def scan_command(
    target: str = typer.Argument(..., help="Host or URL to assess. You must be authorised to test it."),
    authorize: bool = typer.Option(False, "--i-am-authorized", help="Attest that you may test this target."),
    note: Optional[str] = typer.Option(None, "--authorization", "-a", help="Who authorised it, and what is in scope."),
    profile: str = typer.Option("passive", "--profile", help="passive (observe only) or active (bounded probes)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write the scan JSON here."),
    builtin: bool = typer.Option(False, "--builtin", help="Use the built-in crawler even if a real scanner is installed."),
    allow_private: bool = typer.Option(
        False, "--allow-private",
        help="Permit a loopback, private or link-local target, e.g. a container on localhost.",
    ),
    respect_robots: bool = typer.Option(
        False, "--respect-robots",
        help=(
            "Treat robots.txt Disallow entries as boundaries. Off by default: robots.txt is a "
            "crawler convention, not an access control, and honouring it reports paths as clean "
            "that were never looked at. Crawl-delay is honoured either way."
        ),
    ),
    list_tools: bool = typer.Option(False, "--tools", help="List the scanners found on this machine and exit."),
) -> None:
    """Assess a target directly and write a scan, with no scanner report needed."""
    from vulnpriority.scan import ScanProfile, ScanRequest, assess_target
    from vulnpriority.scan.adapters import describe_external_tools
    from vulnpriority.scan.safety import OutOfScopeError, normalise_target

    target_url = normalise_target(target)

    if list_tools:
        typer.echo(describe_external_tools())
        return

    if not authorize or not (note or "").strip():
        _fail(PermissionError(
            "a scan needs an explicit authorisation attestation.\n"
            "  Pass --i-am-authorized and --authorization \"who authorised it, and what is in scope\".\n"
            "  The note is recorded in the report, so the assessment carries its own justification."
        ))
        return

    try:
        request = ScanRequest(
            target_url=target_url, authorized=True, authorization_note=note.strip(),
            profile=ScanProfile(profile), allow_private_targets=allow_private,
            respect_robots=respect_robots,
        )
    except Exception as error:  # noqa: BLE001 - a bad request is a user error, not a crash
        _fail(error)
        return

    if target_url != target:
        typer.echo(f"reading {target!r} as {target_url}")
    typer.echo(f"assessing {target_url} ({profile})")
    try:
        outcome = assess_target(request, force_builtin=builtin,
                                on_progress=lambda event: typer.echo(f"  {getattr(event, 'message', event)}"))
    except OutOfScopeError as error:
        # The safety layer speaks in terms of the Python field it guards. From a terminal
        # that is unactionable advice, so translate it into the flag that actually exists.
        if not allow_private and "private" in str(error).lower():
            _fail(OutOfScopeError(
                f"{target} is a loopback, private or link-local address.\n"
                "  That is refused by default so the scanner cannot be pointed at an internal\n"
                "  network or a cloud metadata endpoint by accident.\n"
                "  If it is your own machine or container, add --allow-private."
            ))
        else:
            _fail(error)
        return
    scan = outcome.scan
    _echo_table("scan", {
        "tool": f"{outcome.tool or 'built-in'} {outcome.tool_version or ''}".strip(),
        "why": outcome.tool_selection or "-",
        "endpoints": len(scan.endpoints),
        "findings": len(scan.findings),
        "pages fetched": outcome.pages_fetched,
        "out of scope skipped": outcome.skipped_out_of_scope,
        "errors": "; ".join(outcome.errors) if outcome.errors else "none",
    })
    # Printed apart from the table and not truncated. A short finding list is ambiguous -
    # it means either a clean target or a scan that could not see the target - and these
    # sentences are the only thing that tells the two apart.
    if outcome.coverage_notes:
        typer.echo("\nread the findings with these in mind:")
        for note in outcome.coverage_notes:
            typer.echo(f"  - {note}")
    destination = Path(output) if output else Path("runs") / f"{scan.scan_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(scan.model_dump(mode="json"), indent=1), encoding="utf-8")
    typer.echo(f"\nscan written to {destination}")
    typer.echo(f"prioritise it with:  vulnpriority run-all --input {destination}")


@app.command("novelty")
def novelty_command(
    as_json: bool = typer.Option(False, "--json", help="Print the whole payload as JSON."),
) -> None:
    """Compare this framework's capabilities against the reviewed prior work."""
    from vulnpriority.novelty import novelty_payload

    payload = novelty_payload()
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    verdict = payload["verdict"]
    typer.echo(verdict["overall_statement"] + "\n")
    for level in ("novel", "novel_combination", "incremental", "not_novel", "not_claimed"):
        items = verdict.get(level) or []
        if not items:
            continue
        typer.echo(f"{level.replace('_', ' ')} ({len(items)})")
        for item in items:
            typer.echo(f"  - {item if isinstance(item, str) else item.get('capability', item)}")
        typer.echo("")
    threats = verdict.get("threats_to_novelty") or []
    typer.echo(f"threats to these claims: {len(threats)} (use --json to read them)")


def main() -> None:
    """Entry point for ``python -m vulnpriority``.

    ``_harden_streams`` also runs from the Typer callback above, which is what covers the
    installed ``vulnpriority`` console script: ``pyproject.toml`` points that at ``app``, not
    here, so an entry point that only hardened in this function would leave the console
    script exactly as broken as it was.
    """
    _harden_streams()
    app()


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    main()
