"""OS-held run lock; automatically released even when a process is killed."""
from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def exclusive_run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / "training.lock", os.O_RDWR | os.O_CREAT)
    with os.fdopen(descriptor, "r+", encoding="ascii") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process owns run directory: {directory}. Stop it before resuming.") from exc
        try:
            handle.seek(0)
            handle.write(f" {os.getpid()}\n")
            handle.truncate()
            handle.flush()
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
