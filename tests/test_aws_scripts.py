import os
import re
import subprocess
from pathlib import Path


def test_scripts_require_env_and_contain_no_host():
    for s in ("sync.sh", "run.sh"):
        p = Path("scripts/aws") / s
        r = subprocess.run(["bash", str(p), "echo"], env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
        assert r.returncode == 2 and "JEFF_TRAIN_" in r.stderr
    for p in Path("scripts/aws").glob("*.sh"):
        t = p.read_text()
        assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", t) and ".pem" not in t


def test_scripts_require_key_when_host_set():
    for s in ("sync.sh", "run.sh"):
        env = {"PATH": os.environ["PATH"], "JEFF_TRAIN_HOST": "u@h"}
        r = subprocess.run(["bash", str(Path("scripts/aws") / s), "echo"], env=env, capture_output=True, text=True)
        assert r.returncode == 2 and "JEFF_TRAIN_KEY" in r.stderr


def _fake_env(tmp_path):
    home = tmp_path / "home"
    (home / "venv/bin").mkdir(parents=True)
    (home / "jeff").mkdir()
    for name, body in (("pip", "exit 0"), ("jeff", 'printf "%s\\n" "$@" > "$HOME/jeff_args"')):
        f = home / "venv/bin" / name
        f.write_text("#!/bin/sh\n" + body + "\n")
        f.chmod(0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    ssh = bindir / "ssh"
    # Record the args, then run the remote command locally with HOME pointed at the fake box.
    ssh.write_text(
        '#!/bin/bash\nfor a in "$@"; do printf "%s\\n" "$a"; done > "$FAKE_SSH_ARGS"\n'
        'HOME="$FAKE_HOME" exec bash -c "${@: -1}"\n'
    )
    ssh.chmod(0o755)
    env = {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "JEFF_TRAIN_HOST": "u@h",
        "JEFF_TRAIN_KEY": "/k/key",
        "FAKE_SSH_ARGS": str(tmp_path / "ssh_args"),
        "FAKE_HOME": str(home),
    }
    return home, env


def test_run_bg_remote_command_and_quoting(tmp_path):
    import time

    home, env = _fake_env(tmp_path)
    r = subprocess.run(
        ["bash", "scripts/aws/run.sh", "--bg", "jeff", "train", "--config", "configs/stage0/lc25.json", "--x", '$HOME "q" `b` \\'],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    remote = (tmp_path / "ssh_args").read_text()
    assert "lc25.log" in remote and "venv/bin" in remote and "/dev/null" in remote
    out = home / "jeff_args"
    for _ in range(50):
        if out.exists() and out.read_text().count("\n") >= 6:
            break
        time.sleep(0.1)
    assert out.read_text().splitlines()[-1] == '$HOME "q" `b` \\'
    assert (home / "runs/lc25.log").exists()


def test_run_bg_explicit_log_name(tmp_path):
    _, env = _fake_env(tmp_path)
    subprocess.run(["bash", "scripts/aws/run.sh", "--bg", "--log", "mine", "jeff", "eval"], env=env, timeout=30)
    assert "mine.log" in (tmp_path / "ssh_args").read_text()


def test_run_foreground_args_unmangled(tmp_path):
    home, env = _fake_env(tmp_path)
    r = subprocess.run(["bash", "scripts/aws/run.sh", "jeff", "a b", "$HOME"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert (home / "jeff_args").read_text().splitlines() == ["a b", "$HOME"]
