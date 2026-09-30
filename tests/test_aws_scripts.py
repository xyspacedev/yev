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
