"""Per-run diagnostic paths, atomic JSON writes and one local job lock."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4


@dataclass(frozen=True)
class RunArtifacts:
    directory: Path

    def __post_init__(self):
        object.__setattr__(self, "directory", Path(self.directory))

    @property
    def run_path(self):
        return self.directory / "run.json"

    @property
    def query_path(self):
        return self.directory / "query.json"

    @property
    def prompt_path(self):
        return self.directory / "llm-prompt.txt"

    @property
    def answer_path(self):
        return self.directory / "llm-answer.txt"


def create_run_artifacts(root):
    """Create a new run directory without reusing or truncating another run."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / f"{stamp}-{uuid4().hex}"
    directory.mkdir(exist_ok=False)
    return RunArtifacts(directory)


def atomic_write_json(path, value):
    """Replace one UTF-8 JSON snapshot, leaving its old version on failure."""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def exclusive_job(lock_path):
    """Reject another local job immediately; the OS releases locks on exit.

    Retain the lock file after release: deleting it can let contenders lock
    different file objects at the same path. The lock is for this local host.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    locked = False
    try:
        # Windows byte-range locking needs a stable byte at offset zero.
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            elif os.name == "posix":
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                raise RuntimeError("Local job locking is unsupported on this platform.")
            locked = True
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise RuntimeError("Another job is already running.") from error
            raise
        yield
    finally:
        try:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
