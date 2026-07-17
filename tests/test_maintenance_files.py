from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_installs_the_documented_test_environment() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert 'pip install -e ".[dev,face,vision,whisper,web,align]"' in workflow


def test_ci_uses_current_action_runtimes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "actions/checkout@v7" in workflow
    assert "actions/setup-python@v6" in workflow
