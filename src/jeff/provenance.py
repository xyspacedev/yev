"""Run provenance: the code revision and training-data hash recorded in metrics files."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHA_FILE = "GIT_SHA"


def git_sha() -> str | None:
    """HEAD from git, else the GIT_SHA file scripts/aws/sync.sh ships (a git archive has no .git)."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=10)
        sha = res.stdout.strip()
        if res.returncode == 0 and sha:
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return (REPO_ROOT / SHA_FILE).read_text().strip() or None
    except OSError:
        return None


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
