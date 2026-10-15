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


def test_engines_ignore_model_generation_config_and_fit_max_len():
    text = Path("scripts/aws/serve_engine.sh").read_text()
    assert "--generation-config vllm" in text
    # yev serve accepts prompts of up to --max-len (16384) tokens; vLLM needs prompt + 1 <= max_model_len
    # and SGLang needs prompt < context_length, so the engines get one more.
    assert 'MAXLEN="${MAXLEN:-16385}"' in text


@pytest.mark.parametrize("path", SCRIPTS[:2])
def test_setup_installs_python_headers(path):
    # Triton compiles a C helper against Python.h when vLLM/SGLang start; the DLAMI has no python3.12-dev.
    assert "python3.12-dev" in Path(path).read_text()


def test_engine_runs_with_its_own_venv_on_path():
    # The engines JIT-build kernels with tools from their venv (ninja); without it on PATH they fail at startup.
    assert 'export PATH="$HOME/venv-$NAME/bin:$PATH"' in Path("scripts/aws/serve_engine.sh").read_text()


def test_engine_takes_extra_args_for_tuning():
    text = Path("scripts/aws/serve_engine.sh").read_text()
    assert 'read -r -a EXTRA <<< "${ENGINE_ARGS:-}"' in text and '"${EXTRA[@]}"' in text
