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
        t = p.read_text().replace("127.0.0.1", "")  # loopback is the server address, not a host
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
    setsid = bindir / "setsid"  # macOS has no setsid; the box (Ubuntu) does
    setsid.write_text('#!/bin/sh\nexec "$@"\n')
    setsid.chmod(0o755)
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
    assert "nohup setsid bash -c" in remote and "&& nohup" not in remote
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
    assert (home / "jeff/GIT_SHA").read_text() == head + "\n"
    (repo / "dirty.txt").write_text("x")  # untracked file makes the tree dirty
    r = subprocess.run(["bash", "scripts/aws/sync.sh"], cwd=repo, env=genv, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert (home / "jeff/GIT_SHA").read_text() == head + "-dirty\n"


def _we_env(tmp_path, health_ok=True, fail_wf=None):
    home = tmp_path / "home"
    (home / "jeff").mkdir(parents=True)
    (home / "jeff/GIT_SHA").write_text("abc123-dirty\n")
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
    assert meta["jeff_git_sha"] == "abc123-dirty"
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
    assert "JEFF_TRAIN" not in t  # runs on the box; needs no laptop-side env
