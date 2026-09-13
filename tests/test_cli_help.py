"""The CLI: every command of DESIGN.md 3.11 is registered, documented and invocable.

Help output is not cosmetic here. The application is the only interface most readers of the
review will ever touch, and three properties have to hold before any of it is useful:

* every command named in the specification exists, with the options that command needs;
* ``--help`` works without importing a pipeline module, so the application stays usable
  while individual packages are still being built and so nothing reaches the network
  merely because a user asked what a command does;
* the one command that has no upstream dependency at all - ``synth`` - really runs, offline,
  and writes a dataset, because ``run-all --synthetic`` is built on it.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vulnpriority import cli
from vulnpriority.cli import app

#: Exactly the command list in DESIGN.md 3.11.
EXPECTED_COMMANDS: tuple[str, ...] = (
    "synth",
    "ingest",
    "assess",
    "enrich",
    "chain",
    "rank",
    "train-ranker",
    "explain",
    "evaluate",
    "ablate",
    "select",
    "simulate",
    "adversarial",
    "report",
    "run-all",
    "fetch-feeds",
    "manifest",
)

#: The application layer of DESIGN.md section 6. Each becomes required once its package
#: lands; until then an absent one is not a failure, but an unlisted command still is.
APPLICATION_COMMANDS: tuple[str, ...] = ("scan", "serve", "web", "novelty")


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


def _registered_names() -> set[str]:
    import typer.main

    command = typer.main.get_command(app)
    return set(getattr(command, "commands", {}))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_every_design_command_is_registered() -> None:
    registered = _registered_names()
    missing = [name for name in EXPECTED_COMMANDS if name not in registered]
    assert missing == [], f"commands missing from the CLI: {missing}"


def test_no_undocumented_commands_were_added() -> None:
    """Every registered command must appear in DESIGN.md; an unlisted one is a drift."""
    documented = set(EXPECTED_COMMANDS) | set(APPLICATION_COMMANDS)
    extra = sorted(_registered_names() - documented)
    assert extra == [], f"commands not in DESIGN.md sections 3.11 or 6: {extra}"


def test_the_application_is_named_vulnpriority() -> None:
    assert app.info.name == "vulnpriority"


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def test_top_level_help_lists_every_command(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output

    # Option and command columns wrap, so compare on whitespace-collapsed text.
    flattened = " ".join(result.output.split())
    for name in EXPECTED_COMMANDS:
        assert name in flattened, f"{name} is not shown in the top-level help"
    assert "vulnerability prioritization" in flattened.lower()


def test_no_arguments_prints_help_rather_than_failing_silently(runner: CliRunner) -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
    for name in ("synth", "run-all"):
        assert name in " ".join(result.output.split())


@pytest.mark.parametrize("command", EXPECTED_COMMANDS)
def test_each_command_has_help(runner: CliRunner, command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
    flattened = " ".join(result.output.split())
    assert f"Usage: {app.info.name} {command}" in flattened or command in flattened
    assert "--help" in flattened


@pytest.mark.parametrize("command", EXPECTED_COMMANDS)
def test_every_stage_command_takes_config_and_output_dir(runner: CliRunner, command: str) -> None:
    """``--config`` and ``--output-dir`` are the two conventions every command honours."""
    result = runner.invoke(app, [command, "--help"])
    flattened = " ".join(result.output.split())
    assert "--config" in flattened, f"{command} has no --config"
    assert "--output-dir" in flattened, f"{command} has no --output-dir"


@pytest.mark.parametrize(
    ("command", "options"),
    [
        ("synth", ("--seed", "--apps", "--scans-per-app", "--injection-fraction")),
        ("ingest", ("--input", "--dataset")),
        ("fetch-feeds", ("--as-of", "--cve")),
        ("enrich", ("--attacker", "--impact")),
        ("rank", ("--ranker", "--attacker", "--top")),
        ("explain", ("--finding",)),
        ("evaluate", ("--split",)),
        ("select", ("--budget-hours", "--method")),
        ("simulate", ("--weeks", "--capacity")),
        ("adversarial", ("--corpus",)),
        ("run-all", ("--synthetic", "--dataset", "--scan", "--keep-going")),
        ("manifest", ("--run-dir", "--json")),
    ],
)
def test_commands_expose_the_options_they_need(
    runner: CliRunner, command: str, options: tuple[str, ...]
) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
    flattened = " ".join(result.output.split())
    missing = [option for option in options if option not in flattened]
    assert missing == [], f"{command} is missing {missing}"


def test_an_unknown_command_fails_cleanly(runner: CliRunner) -> None:
    result = runner.invoke(app, ["definitely-not-a-command"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Behaviour that needs no other package
# ---------------------------------------------------------------------------


def test_synth_generates_a_dataset_offline(runner: CliRunner, tmp_path: Path) -> None:
    """``vulnpriority synth`` is the root of ``run-all --synthetic`` and must work on its own."""
    result = runner.invoke(
        app,
        [
            "synth",
            "--output-dir",
            str(tmp_path),
            "--apps",
            "2",
            "--scans-per-app",
            "2",
            "--seed",
            "5",
            "--set",
            "synthetic.findings_per_app=[6, 10]",
            "--set",
            "synthetic.endpoints_per_app=[12, 13]",
        ],
    )
    assert result.exit_code == 0, result.output

    written = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(written) == 1, f"expected one dataset directory, got {written}"
    root = written[0]

    manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["apps"] == 2
    assert manifest["summary"]["scans"] == 4
    assert manifest["config"]["seed"] == 5

    for relative in (
        "feeds/nvd/cves.json",
        "feeds/epss/epss_snapshots.jsonl",
        "feeds/kev/kev.json",
        "feeds/exploitdb/exploits.jsonl",
        "feeds/references/index.json",
        "oracle.json",
        "world.json",
    ):
        assert (root / relative).is_file(), f"{relative} was not written"
    assert list((root / "scans").glob("*.json"))

    flattened = " ".join(result.output.split())
    assert manifest["dataset_hash"] in flattened
    assert "oracle_positives" in flattened


def test_set_overrides_are_parsed_as_json() -> None:
    from vulnpriority.cli import _parse_overrides

    parsed = _parse_overrides(
        ["ranking.n_estimators=500", "component_c.enabled=false", "llm.model=claude-sonnet-5"]
    )
    assert parsed["ranking.n_estimators"] == 500
    assert parsed["component_c.enabled"] is False
    assert parsed["llm.model"] == "claude-sonnet-5"


def test_set_without_a_value_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["synth", "--output-dir", str(tmp_path), "--set", "nonsense"])
    assert result.exit_code != 0


def test_manifest_on_a_missing_run_reports_the_directory(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["manifest", "--run-dir", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "no manifest" in " ".join(result.output.split()).lower()


def test_run_all_without_a_source_explains_itself(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["run-all", "--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    flattened = " ".join(result.output.split()).lower()
    assert "--synthetic" in flattened and "--scan" in flattened


def test_help_does_not_import_the_pipeline(runner: CliRunner) -> None:
    """Importing the CLI must stay cheap: nothing downstream is loaded to print help.

    This is what keeps the application usable while ``graph``, ``rank``, ``eval`` and
    ``adversarial`` are still being written, and it is what makes the offline guarantee
    structural rather than a matter of discipline.
    """
    import subprocess
    import sys

    probe = (
        "import sys; from vulnpriority.cli import app;"
        "loaded=[name for name in sys.modules if name.startswith('vulnpriority.')];"
        "print(','.join(sorted(loaded)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    loaded = set(completed.stdout.strip().split(","))
    for forbidden in (
        "vulnpriority.pipeline",
        "vulnpriority.pipeline.runner",
        "vulnpriority.synth.generator",
        "vulnpriority.rank",
        "vulnpriority.eval",
        "vulnpriority.graph",
    ):
        assert forbidden not in loaded, f"importing the CLI pulled in {forbidden}"


# ---------------------------------------------------------------------------
# The console's encoding
#
# The money figures are rupees, and U+20B9 is not in cp1252 -- which is the stream a
# Windows console hands Python by default. Writing one raised UnicodeEncodeError from
# inside ``typer.echo`` and aborted ``run-all`` at the budget table, after the whole
# pipeline had already succeeded.
#
# The headline case runs in a subprocess on a genuinely cp1252 stdout, because that is the
# only way to exercise the real stream: pytest owns ``sys.stdout`` inside a test and hands
# it back between phases, so patching it in-process tests something else.
# ---------------------------------------------------------------------------


CONSOLE_PROBE = """
import sys
from vulnpriority import cli
cli._harden_streams()
import typer
typer.echo("risk captured " + cli._money(195636364, "INR"))
typer.echo("total " + cli._money_phrase(195636364, "INR"))
typer.echo("usd run " + cli._money(195636364, "USD"))
typer.echo("reason code: expected loss ₹50,000 = P(exploit) 0.25")
"""


def _run_on_console(encoding: str) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ, PYTHONIOENCODING=encoding)
    return subprocess.run(
        [sys.executable, "-c", CONSOLE_PROBE],
        capture_output=True,
        env=env,
        encoding=encoding,
        errors="replace",
    )


def test_printing_money_on_a_legacy_console_does_not_abort_the_command() -> None:
    """The regression: before the fix this died with UnicodeEncodeError, exit code 1."""
    done = _run_on_console("cp1252")

    assert done.returncode == 0, done.stderr
    assert "UnicodeEncodeError" not in done.stderr
    assert "risk captured Rs 19,56,36,364" in done.stdout
    assert "total Rs 19.56 crore" in done.stdout


def test_a_legacy_console_is_not_given_a_symbol_it_cannot_draw() -> None:
    """No mojibake and no bare question mark: the amount keeps a readable currency."""
    done = _run_on_console("cp1252")

    assert done.returncode == 0, done.stderr
    assert "₹" not in done.stdout
    assert "?" not in done.stdout
    # Even a symbol this module did not format itself is spelled out rather than lost.
    assert "expected loss Rs 50,000 = P(exploit) 0.25" in done.stdout


def test_a_currency_the_console_can_encode_is_left_alone() -> None:
    """``$`` is in cp1252, so a USD run on a legacy console is not degraded."""
    done = _run_on_console("cp1252")

    assert done.returncode == 0, done.stderr
    assert "usd run $195,636,364" in done.stdout


def test_a_utf8_console_keeps_the_real_symbol() -> None:
    """The fallback is for consoles that need it, not a downgrade for everybody."""
    done = _run_on_console("utf-8")

    assert done.returncode == 0, done.stderr
    assert "risk captured ₹19,56,36,364" in done.stdout
    assert "total ₹19.56 crore" in done.stdout


def test_the_console_spelling_is_chosen_per_currency(monkeypatch: pytest.MonkeyPatch) -> None:
    """The decision itself is a pure function of the stream's declared encoding."""
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    assert cli._ascii_console("INR") is True
    assert cli._ascii_console("USD") is False

    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))
    assert cli._ascii_console("INR") is False


def test_hardening_a_stream_that_cannot_be_reconfigured_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrapped or redirected stream has no ``reconfigure``; that is not a reason to fail."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    cli._harden_streams()
