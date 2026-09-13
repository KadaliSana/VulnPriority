"""Secrets loading: the file is a convenience, the environment is the authority.

A secrets loader is a small thing that goes wrong in expensive ways. The three properties
that matter are pinned here: a real environment variable is never shadowed by a stale file,
a key never reaches the config hash or the run manifest, and a malformed file degrades
rather than raising in the middle of someone's scan.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vulnpriority.core.config import DOTENV_PATH, PROJECT_ROOT, load_config, load_dotenv


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in ("VULNPRIORITY_TEST_KEY", "VULNPRIORITY_TEST_OTHER", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_values_reach_the_environment(clean_env, tmp_path: Path) -> None:
    load_dotenv(_write(tmp_path, "VULNPRIORITY_TEST_KEY=abc123\n"))
    assert os.environ["VULNPRIORITY_TEST_KEY"] == "abc123"


def test_a_real_environment_variable_wins(clean_env, tmp_path: Path) -> None:
    """The precedence people expect: an exported key or a CI secret is not overwritten."""
    clean_env.setenv("VULNPRIORITY_TEST_KEY", "from-the-environment")
    load_dotenv(_write(tmp_path, "VULNPRIORITY_TEST_KEY=from-the-file\n"))
    assert os.environ["VULNPRIORITY_TEST_KEY"] == "from-the-environment"


def test_override_is_available_but_not_the_default(clean_env, tmp_path: Path) -> None:
    clean_env.setenv("VULNPRIORITY_TEST_KEY", "from-the-environment")
    load_dotenv(_write(tmp_path, "VULNPRIORITY_TEST_KEY=from-the-file\n"), override=True)
    assert os.environ["VULNPRIORITY_TEST_KEY"] == "from-the-file"


@pytest.mark.parametrize(
    "line,expected",
    [
        ("VULNPRIORITY_TEST_KEY=plain", "plain"),
        ('VULNPRIORITY_TEST_KEY="double quoted"', "double quoted"),
        ("VULNPRIORITY_TEST_KEY='single quoted'", "single quoted"),
        ("  VULNPRIORITY_TEST_KEY = spaced  ", "spaced"),
        ("export VULNPRIORITY_TEST_KEY=exported", "exported"),
        ("VULNPRIORITY_TEST_KEY=sk-ant-api03-aa/bb+cc=", "sk-ant-api03-aa/bb+cc="),
    ],
)
def test_line_forms(clean_env, tmp_path: Path, line: str, expected: str) -> None:
    load_dotenv(_write(tmp_path, line + "\n"))
    assert os.environ["VULNPRIORITY_TEST_KEY"] == expected


def test_comments_and_blank_lines_are_ignored(clean_env, tmp_path: Path) -> None:
    load_dotenv(_write(tmp_path, "# a comment\n\n   \nVULNPRIORITY_TEST_KEY=value\n"))
    assert os.environ["VULNPRIORITY_TEST_KEY"] == "value"
    assert "# a comment" not in os.environ


def test_a_malformed_file_degrades_rather_than_raising(clean_env, tmp_path: Path) -> None:
    """A broken secrets file must not abort a scan that did not need a key."""
    load_dotenv(_write(tmp_path, "not a key value line\n=missing-name\nVULNPRIORITY_TEST_KEY=ok\n"))
    assert os.environ["VULNPRIORITY_TEST_KEY"] == "ok"


def test_a_missing_file_is_not_an_error(clean_env, tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "does-not-exist") == {}


def test_the_return_value_never_carries_the_secret(clean_env, tmp_path: Path) -> None:
    applied = load_dotenv(_write(tmp_path, "VULNPRIORITY_TEST_KEY=super-secret-value\n"))
    assert "VULNPRIORITY_TEST_KEY" in applied
    assert "super-secret-value" not in json.dumps(applied)


def test_a_key_never_reaches_the_config_hash_or_the_manifest(clean_env, tmp_path: Path) -> None:
    """The config records the variable NAME to read, never the value behind it."""
    before = load_config(PROJECT_ROOT / "configs" / "default.yaml").hash()
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-appear")
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

    assert config.hash() == before, "a secret changed the config hash"
    dumped = json.dumps(config.model_dump(mode="json"))
    assert "sk-ant-should-never-appear" not in dumped
    assert config.llm.api_key_env == "ANTHROPIC_API_KEY"


def test_the_committed_template_exists_and_holds_no_value() -> None:
    example = PROJECT_ROOT / ".env.example"
    assert example.exists(), ".env.example is the documented way to set a key"
    for line in example.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition("=")
        assert value == "", f"{name} carries a value in the committed template"


def test_the_real_file_is_ignored_by_git() -> None:
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignore
    assert "!.env.example" in ignore


def test_the_default_path_is_the_project_root() -> None:
    assert DOTENV_PATH == PROJECT_ROOT / ".env"


def test_a_blank_assignment_sets_nothing(clean_env, tmp_path: Path) -> None:
    """A placeholder must not resolve to a provider that cannot authenticate.

    Backend selection asks whether a key is *present*. If a blank line in the template set
    the variable to an empty string, an untouched file would make every provider look
    configured and the live path would be chosen with no credentials behind it.
    """
    load_dotenv(_write(tmp_path, "VULNPRIORITY_TEST_KEY=\nVULNPRIORITY_TEST_OTHER=real\n"))
    assert "VULNPRIORITY_TEST_KEY" not in os.environ
    assert os.environ["VULNPRIORITY_TEST_OTHER"] == "real"


def test_the_working_env_file_is_readable_and_never_committed(clean_env) -> None:
    """``.env`` is where a real key belongs, so it may hold values -- but it must be ignored.

    An earlier version of this test asserted the file was all placeholders. That was wrong:
    ``.env`` exists precisely so that a key can live somewhere git will not take it, and
    asserting it stays empty would fail the moment the file was used as intended. What must
    be true is that it parses, that nothing it contains reaches a committed file, and that
    the exclusion in ``.gitignore`` is real -- which the two tests either side of this cover.
    """
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        pytest.skip("no .env in this checkout")

    for line in env.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "=" in stripped, f"unparseable line in .env: {stripped[:40]!r}"

    # Whatever it holds, no value from it may appear in the committed template.
    example_values = [
        line.partition("=")[2].strip()
        for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    ]
    assert not any(example_values), "the committed template carries a value"
