"""CWD-independent imports and default paths for Buck analysis scripts."""

from pathlib import Path
import sys

BUCK_ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    REPO_ROOT = BUCK_ROOT.parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = str(BUCK_ROOT / "configs" / "hybrid_event_matlab_v3" / "project.json")
DEFAULT_DATA_DIR = str(BUCK_ROOT / "data" / "processed" / "hybrid_event_matlab_v3")
DEFAULT_CKPT = str(BUCK_ROOT / "artifacts" / "checkpoint.pt")
DEFAULT_WEIGHTS = str(BUCK_ROOT / "artifacts" / "best.pt")
DEFAULT_ARTIFACTS_DIR = str(BUCK_ROOT / "artifacts")
DEFAULT_WAVEFORM_DIR = str(BUCK_ROOT / "artifacts" / "waveforms")
