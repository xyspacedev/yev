"""Sidecar engine scripts parse and carry no host details (Plan 7 Task 4)."""

import subprocess
from pathlib import Path

import pytest

SCRIPTS = ["scripts/aws/setup_vllm.sh", "scripts/aws/setup_sglang.sh", "scripts/aws/serve_engine.sh"]


@pytest.mark.parametrize("path", SCRIPTS)
def test_engine_scripts_parse(path):
    assert Path(path).exists(), path
    subprocess.run(["bash", "-n", path], check=True)


def test_no_host_in_engine_scripts():
    for path in SCRIPTS:
        text = Path(path).read_text()
        assert "amazonaws" not in text and ".pem" not in text and "i-0" not in text
