import os
import re
import subprocess
from pathlib import Path


def test_scripts_require_env_and_contain_no_host():
    for s in ("sync.sh", "run.sh"):
        p = Path("scripts/aws") / s
        r = subprocess.run(["bash", str(p), "echo"], env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
        assert r.returncode == 2 and "YEV_TRAIN_" in r.stderr
    for p in Path("scripts/aws").glob("*.sh"):
        t = p.read_text().replace("127.0.0.1", "")  # loopback is the server address, not a host
        assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", t) and ".pem" not in t


def test_scripts_require_key_when_host_set():
    for s in ("sync.sh", "run.sh"):
        env = {"PATH": os.environ["PATH"], "YEV_TRAIN_HOST": "u@h"}
        r = subprocess.run(["bash", str(Path("scripts/aws") / s), "echo"], env=env, capture_output=True, text=True)
        assert r.returncode == 2 and "YEV_TRAIN_KEY" in r.stderr


def _fake_env(tmp_path):
    home = tmp_path / "home"
    (home / "venv/bin").mkdir(parents=True)
    (home / "yev").mkdir()
    for name, body in (("pip", "exit 0"), ("yev", 'printf "%s\\n" "$@" > "$HOME/yev_args"')):
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
    setsid = bindir / "setsid"  # macOS has no setsid; the box (Ubuntu) does
    setsid.write_text('#!/bin/sh\nexec "$@"\n')
    setsid.chmod(0o755)
    env = {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "YEV_TRAIN_HOST": "u@h",
        "YEV_TRAIN_KEY": "/k/key",
        "FAKE_SSH_ARGS": str(tmp_path / "ssh_args"),
        "FAKE_HOME": str(home),
    }
    return home, env


def test_run_bg_remote_command_and_quoting(tmp_path):
    import time

    home, env = _fake_env(tmp_path)
    r = subprocess.run(
        ["bash", "scripts/aws/run.sh", "--bg", "yev", "train", "--config", "configs/stage0/lc25.json", "--x", '$HOME "q" `b` \\'],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    remote = (tmp_path / "ssh_args").read_text()
    assert "lc25.log" in remote and "venv/bin" in remote and "/dev/null" in remote
    assert "nohup setsid bash -c" in remote and "&& nohup" not in remote
    out = home / "yev_args"
    for _ in range(50):
        if out.exists() and out.read_text().count("\n") >= 6:
            break
        time.sleep(0.1)
    assert out.read_text().splitlines()[-1] == '$HOME "q" `b` \\'
    assert (home / "runs/lc25.log").exists()


def test_run_bg_explicit_log_name(tmp_path):
    _, env = _fake_env(tmp_path)
    subprocess.run(["bash", "scripts/aws/run.sh", "--bg", "--log", "mine", "yev", "eval"], env=env, timeout=30)
    assert "mine.log" in (tmp_path / "ssh_args").read_text()


def test_run_foreground_args_unmangled(tmp_path):
    home, env = _fake_env(tmp_path)
    r = subprocess.run(["bash", "scripts/aws/run.sh", "yev", "a b", "$HOME"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert (home / "yev_args").read_text().splitlines() == ["a b", "$HOME"]


def test_run_bg_returns_before_job_finishes(tmp_path):
    import time

    home, env = _fake_env(tmp_path)
    t0 = time.time()
    # capture_output waits for EOF on the pipes: it hangs if the background job still holds ssh's stdout.
    r = subprocess.run(["bash", "scripts/aws/run.sh", "--bg", "--log", "slow", "sleep", "5"],
                       env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert time.time() - t0 < 3


def test_sync_pushes_only_mix_dev_examples_and_bench():
    t = (Path("scripts/aws") / "sync.sh").read_text()
    sources = re.findall(r"^\s*rsync .*\"\" (data/\S*) ", t, re.M)
    assert sources == ["data/mix/stage0/", "data/dev/decidebench_examples.jsonl", "data/bench/"]


def _sync_repo(tmp_path, env):
    repo = tmp_path / "repo"
    (repo / "scripts/aws").mkdir(parents=True)
    (repo / "scripts/aws/sync.sh").write_text(Path("scripts/aws/sync.sh").read_text())
    (repo / "data/mix/stage0").mkdir(parents=True)
    (repo / "data/mix/stage0/x").write_text("x")
    (repo / "data/dev").mkdir()
    (repo / "data/dev/decidebench_examples.jsonl").write_text("{}")
    rsync = tmp_path / "bin/rsync"
    rsync.write_text("#!/bin/sh\nexit 0\n")
    rsync.chmod(0o755)
    genv = {**env, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": "/dev/null"}
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "c"]):
        subprocess.run(["git", *cmd], cwd=repo, env=genv, check=True)
    return repo, genv


def test_sync_records_git_sha_on_box(tmp_path):
    home, env = _fake_env(tmp_path)
    repo, genv = _sync_repo(tmp_path, env)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    r = subprocess.run(["bash", "scripts/aws/sync.sh"], cwd=repo, env=genv, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert (home / "yev/GIT_SHA").read_text() == head + "\n"
    (repo / "dirty.txt").write_text("x")  # untracked file makes the tree dirty
    r = subprocess.run(["bash", "scripts/aws/sync.sh"], cwd=repo, env=genv, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert (home / "yev/GIT_SHA").read_text() == head + "-dirty\n"


def _we_env(tmp_path, health_ok=True, fail_wf=None):
    home = tmp_path / "home"
    (home / "yev").mkdir(parents=True)
    (home / "yev/GIT_SHA").write_text("abc123-dirty\n")
    we = tmp_path / "we"
    we.mkdir()
    runner = tmp_path / "fakerunner"
    runner.write_text(
        '#!/bin/bash\nmkdir -p runs/$1/yev-4b\necho "{\\"wf\\": \\"$1\\"}" > runs/$1/yev-4b/scores.json\n'
        'echo "{}" > runs/$1/yev-4b/results.json\n'
        'echo "$TYPESAFE_API_KEY $@" >> "$HOME/runner_calls"\n'
        f'[ "$1" = "{fail_wf}" ] && exit 1\nexit 0\n'
    )
    runner.chmod(0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    curl = bindir / "curl"
    curl.write_text(f'#!/bin/sh\necho "$@" > "$HOME/curl_args"\nexit {0 if health_ok else 22}\n')
    curl.chmod(0o755)
    env = {"PATH": f"{bindir}:{os.environ['PATH']}", "HOME": str(home), "WE_DIR": str(we), "WE_RUNNER": str(runner)}
    return home, env


WF4 = ["invoice_processing", "customer_service", "agent_trace_observability", "security_incidents"]


def test_workflowevals_runs_four_workflows_and_collects(tmp_path):
    import json

    home, env = _we_env(tmp_path)
    r = subprocess.run(["bash", "scripts/aws/workflowevals.sh"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "http://127.0.0.1:8000/health" in (home / "curl_args").read_text()
    calls = (home / "runner_calls").read_text().splitlines()
    assert [c.split()[1] for c in calls] == WF4
    for c in calls:
        assert c.split()[0] == "local"
        assert c.endswith("--model typesafe:jev-1.13.0 --base-url http://127.0.0.1:8000 --name yev-4b")
    out = home / "workflowevals/yev-4b"
    for wf in WF4:
        assert (out / f"{wf}.log").exists()
        assert json.loads((out / wf / "scores.json").read_text()) == {"wf": wf}
        assert (out / wf / "results.json").exists()
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["yev_git_sha"] == "abc123-dirty"
    assert meta["workflowevals_commit"] == "unknown"  # the fake WE_DIR is not a git checkout
    assert list(meta["workflows"]) == WF4
    assert all(isinstance(v["wall_clock_s"], int) and v["exit_code"] == 0 for v in meta["workflows"].values())


def test_workflowevals_fails_fast_without_server(tmp_path):
    home, env = _we_env(tmp_path, health_ok=False)
    r = subprocess.run(["bash", "scripts/aws/workflowevals.sh"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "/health" in r.stderr
    assert not (home / "runner_calls").exists()


def test_workflowevals_continues_after_a_failed_workflow(tmp_path):
    import json

    home, env = _we_env(tmp_path, fail_wf="customer_service")
    r = subprocess.run(["bash", "scripts/aws/workflowevals.sh"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    meta = json.loads((home / "workflowevals/yev-4b/run_meta.json").read_text())
    assert meta["workflows"]["customer_service"]["exit_code"] == 1
    assert meta["workflows"]["security_incidents"]["exit_code"] == 0


def test_workflowevals_pins_commit_and_has_no_host():
    t = Path("scripts/aws/workflowevals.sh").read_text()
    assert re.search(r"WE_COMMIT:-[0-9a-f]{40}\}", t)
    assert "YEV_TRAIN" not in t  # runs on the box; needs no laptop-side env


def _we_default_env(tmp_path, head=None, with_uv=True, existing=True):
    """Default-runner harness: fake git/uv/pip/curl on PATH log every call to $HOME/calls."""
    home = tmp_path / "home"
    (home / "yev").mkdir(parents=True)
    (home / "venv/bin").mkdir(parents=True)
    (home / "yev/GIT_SHA").write_text("abc\n")
    commit = "a" * 40
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv_body = (
        '#!/bin/bash\necho "uv $*" >> "$HOME/calls"\n'
        '[ "$1" = run ] || exit 0\nshift 3\n'  # uv run python run.py <wf> ...
        'mkdir -p runs/$1/yev-4b\necho "{\\"fresh\\": true}" > runs/$1/yev-4b/scores.json\nexit 0\n'
    )
    scripts = {
        "git": (
            '#!/bin/bash\necho "git $*" >> "$HOME/calls"\n'
            'case "$*" in clone*) mkdir -p "${@: -1}"; touch "${@: -1}/run.py";; '
            f'*rev-parse*) echo "{head or commit}";; esac\nexit 0\n'
        ),
        "curl": "#!/bin/sh\nexit 0\n",
    }
    for name, body in scripts.items():
        (bindir / name).write_text(body)
    if with_uv:
        (bindir / "uv").write_text(uv_body)
    else:
        # pip "installs" uv into the venv; the script must call it
        (home / "venv/bin/pip").write_text(
            '#!/bin/bash\necho "pip $*" >> "$HOME/calls"\n'
            f"cat > \"$HOME/venv/bin/uv\" <<'EOF'\n{uv_body}EOF\nchmod +x \"$HOME/venv/bin/uv\"\n"
        )
        (home / "venv/bin/pip").chmod(0o755)
    for f in bindir.iterdir():
        f.chmod(0o755)
    we = tmp_path / "we"
    if existing:
        we.mkdir()
        (we / "run.py").write_text("")
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(home),
        "WE_DIR": str(we),
        "WE_COMMIT": commit,
        "WE_RUNNER": "uv run python run.py",
    }
    return home, we, env


def _run_we(env, **extra):
    return subprocess.run(["bash", "scripts/aws/workflowevals.sh"], env={**env, **extra}, capture_output=True, text=True, timeout=60)


def test_workflowevals_default_runner_checks_out_pin_and_always_syncs(tmp_path):
    home, we, env = _we_default_env(tmp_path)  # existing clone
    r = _run_we(env)
    assert r.returncode == 0, r.stderr
    calls = (home / "calls").read_text().splitlines()
    assert f"git -C {we} fetch -q origin" in calls
    assert f"git -C {we} checkout -q {'a' * 40}" in calls
    assert "uv sync --locked" in calls  # outside the clone branch
    assert not any("clone" in c for c in calls)
    assert calls.index("uv sync --locked") > calls.index(f"git -C {we} checkout -q {'a' * 40}")
    assert sum(c.startswith("uv run") for c in calls) == 4


def test_workflowevals_fresh_clone_then_sync(tmp_path):
    home, we, env = _we_default_env(tmp_path, existing=False)
    r = _run_we(env)
    assert r.returncode == 0, r.stderr
    calls = (home / "calls").read_text().splitlines()
    assert any(c.startswith("git clone") for c in calls) and "uv sync --locked" in calls


def test_workflowevals_installs_uv_when_missing(tmp_path):
    home, we, env = _we_default_env(tmp_path, with_uv=False)
    r = _run_we(env)
    assert r.returncode == 0, r.stderr
    calls = (home / "calls").read_text().splitlines()
    assert calls[0] == "pip install uv"
    assert "uv sync --locked" in calls


def test_workflowevals_exit_4_on_pin_mismatch(tmp_path):
    home, we, env = _we_default_env(tmp_path, head="b" * 40)
    r = _run_we(env)
    assert r.returncode == 4 and "is not" in r.stderr
    calls = (home / "calls").read_text()
    assert "uv sync" not in calls and "uv run" not in calls
    assert not (home / "workflowevals/yev-4b/run_meta.json").exists()


def test_workflowevals_removes_stale_results_unless_resume(tmp_path):
    for resume in (False, True):
        sub = tmp_path / ("resume" if resume else "fresh")
        sub.mkdir()
        home, we, env = _we_default_env(sub)
        # fake uv run only writes scores.json, so a surviving results.json or marker is stale
        for d in (we / "runs/customer_service/yev-4b", home / "workflowevals/yev-4b/customer_service"):
            d.mkdir(parents=True)
            (d / "results.json").write_text("stale")
        r = _run_we(env, **({"WE_RESUME": "1"} if resume else {}))
        assert r.returncode == 0, r.stderr
        stale_src = (we / "runs/customer_service/yev-4b/results.json").exists()
        stale_dst = (home / "workflowevals/yev-4b/customer_service/results.json").exists()
        assert stale_src is resume and stale_dst is resume
        assert (home / "workflowevals/yev-4b/customer_service/scores.json").exists()
