import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ci_installs_the_documented_test_environment() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert 'pip install -e ".[dev,face,vision,whisper,web,align]"' in workflow


def test_ci_uses_current_action_runtimes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "actions/checkout@v7" in workflow
    assert "actions/setup-python@v6" in workflow


def test_pre_push_hook_runs_the_content_guard(tmp_path: Path) -> None:
    hook = ROOT / ".githooks" / "pre-push"
    fake_brigade = tmp_path / "brigade"
    args_file = tmp_path / "args"
    fake_brigade.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$GUARD_ARGS_FILE"\n')
    fake_brigade.chmod(0o755)
    env = {
        "GUARD_ARGS_FILE": str(args_file),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }

    result = subprocess.run([hook], cwd=ROOT, env=env, capture_output=True, text=True)

    assert hook.stat().st_mode & 0o111
    assert result.returncode == 0, result.stderr
    assert args_file.read_text().splitlines() == [
        "guard",
        "git",
        "--history",
        "--range",
        "origin/main..HEAD",
    ]


def test_pre_push_hook_fails_when_brigade_is_unavailable(tmp_path: Path) -> None:
    hook = ROOT / ".githooks" / "pre-push"
    env = {"PATH": str(tmp_path)}

    result = subprocess.run([hook], cwd=ROOT, env=env, capture_output=True, text=True)

    assert result.returncode == 1
    assert "brigade is required" in result.stderr


def test_readme_enables_the_tracked_git_hooks() -> None:
    readme = (ROOT / "README.md").read_text()

    assert "git config core.hooksPath .githooks" in readme
