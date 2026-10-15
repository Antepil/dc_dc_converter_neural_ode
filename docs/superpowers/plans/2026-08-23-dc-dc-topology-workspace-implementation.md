# DC–DC Topology Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent `dc_dc_converter_neural_ode/` workspace, copy the complete Buck project into it, and adapt the copied project so data generation, training, resume, evaluation, and analysis run entirely from the new layout without modifying legacy content.

**Architecture:** Each converter topology is a self-contained project under one of four stable topology families. The first implementation is `topologies/non_isolated_unidirectional/buck/`; no shared framework, registry, unified CLI, empty topology implementations, or rectifier component is created. Legacy `training/` and `neural_ode_v2/` remain untouched, while meaningful Buck code, documentation, data, and artifacts are copied into the new workspace.

**Tech Stack:** Python 3, PyTorch, torchdiffeq 0.2.5, NumPy, SciPy, Matplotlib, MATLAB/Simulink, `unittest`, Git, rsync/shasum.

---

## File map

### Workspace-level files

- Create: `dc_dc_converter_neural_ode/.gitignore` — ignore copied datasets, artifacts, caches, and temporary migration evidence.
- Create: `dc_dc_converter_neural_ode/README.md` — workspace purpose, implemented topology, and commands.
- Create: `dc_dc_converter_neural_ode/AGENTS.md` — local constraints for all future topology work.
- Create: `dc_dc_converter_neural_ode/docs/architecture.md` — self-contained topology architecture and dependency rules.
- Create: `dc_dc_converter_neural_ode/docs/topology_catalog.md` — planned taxonomy versus implemented status.

### Buck project files

- Create from copy: `dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py` — Buck CLI entry adapted from `training/train_buck.py`.
- Create from copy: `.../buck/requirements.txt` — runtime requirements plus SciPy for MAT conversion.
- Create: `.../buck/training/__init__.py` — training package marker.
- Create from copy: `.../buck/training/{data.py,model.py,loops.py,losses.py,evaluate.py,checkpoint.py}` — existing runtime modules; `loops.py` comes from legacy `training/train.py`.
- Create: `.../buck/analysis/__init__.py` — analysis package marker.
- Create: `.../buck/analysis/_bootstrap.py` — CWD-independent Buck root and default path resolution.
- Create from copy: `.../buck/analysis/{plot_extra_hi.py,plot_ripple.py,plot_ripple_vc.py,plot_waveforms.py}` — copied and path-adapted analysis tools.
- Create: `.../buck/tests/__init__.py` — test package marker.
- Copy: `.../buck/configs/` — existing v2.1.1 config.
- Copy: `.../buck/matlab/` — Simulink model and MATLAB scripts, excluding caches.
- Copy: `.../buck/scripts/` — MAT-to-NPZ converter.
- Copy: `.../buck/data/` — all raw, processed, CSV, archive, and manifest content.
- Copy: `.../buck/artifacts/` — all checkpoints, weights, metrics, and plots.
- Create from copies: `.../buck/docs/{ROUTE.md,requirements-alignment.md,CHANGE_LOG.md}` — historical Buck design records.
- Create: `.../buck/README.md` — current local commands and data flow.

### Tests

- Create: `.../buck/tests/test_project_paths.py` — entry imports, defaults, CWD independence, and no legacy runtime references.
- Create: `.../buck/tests/test_analysis_paths.py` — analysis imports/defaults and no legacy runtime references.
- Create: `.../buck/tests/test_generation_layout.py` — MATLAB/converter path assumptions and SciPy dependency.
- Create: `.../buck/tests/test_regression_equivalence.py` — copied NPZ, normalized data, windows, state dict, and rollout equivalence.
- Create: `.../buck/tests/test_documentation.py` — documentation reflects actual implementation status and paths.

## Execution constraints

- Execute in the current repository working tree because the source `neural_ode_v2/` contains untracked data that is absent from a clean Git worktree.
- Stage and commit only paths under `dc_dc_converter_neural_ode/`.
- Never run `git restore`, `git checkout --`, `git clean`, or a broad `git add -A`.
- Do not edit any existing path outside `dc_dc_converter_neural_ode/`.
- Do not overwrite the original or copied `checkpoint.pt` during smoke tests; use temporary checkpoint and output directories.

---

### Task 1: Establish preservation guardrails and ignore rules

**Files:**
- Create: `dc_dc_converter_neural_ode/.gitignore`
- Create at runtime, ignored: `dc_dc_converter_neural_ode/.migration/legacy-content.sha256`
- Create at runtime, ignored: `dc_dc_converter_neural_ode/.migration/legacy-paths.before`
- Create at runtime, ignored: `dc_dc_converter_neural_ode/.migration/git-status.before`

- [ ] **Step 1: Create workspace ignore rules before copying large data**

Create `dc_dc_converter_neural_ode/.gitignore` with exactly:

```gitignore
# Python/macOS caches
**/.DS_Store
**/__pycache__/
**/*.pyc

# MATLAB/Simulink caches
**/*.slxc
**/slprj/

# Full copied datasets and training outputs stay local
**/data/
**/artifacts/
**/train.log

# One-time migration verification evidence
.migration/
```

- [ ] **Step 2: Verify ignore rules before copying**

Run:

```bash
mkdir -p dc_dc_converter_neural_ode/.migration
mkdir -p dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/{data,artifacts}
git check-ignore -v \
  dc_dc_converter_neural_ode/.migration/probe \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/data/probe.npz \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/checkpoint.pt
```

Expected: three lines, each attributed to `dc_dc_converter_neural_ode/.gitignore`.

- [ ] **Step 3: Capture legacy Git status, content hashes, and path inventory**

Run from repository root:

```bash
set -e
MIGRATION=dc_dc_converter_neural_ode/.migration

git status --porcelain=v1 --untracked-files=all \
  | grep -v '^?? dc_dc_converter_neural_ode/' \
  > "$MIGRATION/git-status.before"

{
  find training neural_ode_v2 -type f \
    ! -name '.DS_Store' ! -name '*.pyc' ! -name '*.slxc' \
    ! -path '*/__pycache__/*' ! -path '*/slprj/*' -print0
  printf '%s\0' AGENTS.md ROUTE.md requirements-alignment.md train.log
} | xargs -0 shasum -a 256 > "$MIGRATION/legacy-content.sha256"

{
  find training neural_ode_v2 -type f \
    ! -name '.DS_Store' ! -name '*.pyc' ! -name '*.slxc' \
    ! -path '*/__pycache__/*' ! -path '*/slprj/*' -print0
  printf '%s\0' AGENTS.md ROUTE.md requirements-alignment.md train.log
} | LC_ALL=C sort -z | tr '\0' '\n' > "$MIGRATION/legacy-paths.before"

test "$(wc -l < "$MIGRATION/legacy-paths.before")" -eq 532
test "$(LC_ALL=C sort -u "$MIGRATION/legacy-paths.before" | wc -l)" -eq 532
shasum -a 256 -c "$MIGRATION/legacy-content.sha256" >/dev/null
```

Expected: exit code 0; `legacy-paths.before` contains 532 sorted, unique paths. Do not continue if any listed legacy file is missing or changes during execution.

- [ ] **Step 4: Commit guardrails only**

```bash
git add dc_dc_converter_neural_ode/.gitignore
git diff --cached --check
git commit -m "chore: guard multi-topology workspace outputs"
```

Expected: commit contains only `.gitignore`.

---

### Task 2: Copy the complete Buck project without changing the source

**Files:**
- Create from existing files: all Buck files listed in the file map
- Do not modify: `training/**`, `neural_ode_v2/**`, `AGENTS.md`, `ROUTE.md`, `requirements-alignment.md`, `train.log`

- [ ] **Step 1: Create target directories and copy Python source**

Run:

```bash
set -e
BUCK=dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck
mkdir -p "$BUCK"/{training,analysis,configs,matlab,scripts,docs,data,artifacts,tests}

cp training/train_buck.py "$BUCK/train.py"
cp training/requirements.txt "$BUCK/requirements.txt"
cp training/data.py "$BUCK/training/data.py"
cp training/model.py "$BUCK/training/model.py"
cp training/train.py "$BUCK/training/loops.py"
cp training/losses.py "$BUCK/training/losses.py"
cp training/evaluate.py "$BUCK/training/evaluate.py"
cp training/checkpoint.py "$BUCK/training/checkpoint.py"
cp training/plot_extra_hi.py "$BUCK/analysis/plot_extra_hi.py"
cp training/plot_ripple.py "$BUCK/analysis/plot_ripple.py"
cp training/plot_ripple_vc.py "$BUCK/analysis/plot_ripple_vc.py"
cp training/plot_waveforms.py "$BUCK/analysis/plot_waveforms.py"

printf '%s\n' '"""Buck training package."""' > "$BUCK/training/__init__.py"
printf '%s\n' '"""Buck post-training analysis package."""' > "$BUCK/analysis/__init__.py"
printf '%s\n' '"""Buck workspace tests."""' > "$BUCK/tests/__init__.py"
```

Expected: all listed files exist; legacy files still hash-check successfully.

- [ ] **Step 2: Copy configuration, MATLAB, conversion scripts, data, and artifacts**

Run:

```bash
set -e
BUCK=dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck
EXCLUDES=(--exclude='.DS_Store' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.slxc' --exclude='slprj')

rsync -a "${EXCLUDES[@]}" neural_ode_v2/configs/ "$BUCK/configs/"
rsync -a "${EXCLUDES[@]}" neural_ode_v2/matlab/ "$BUCK/matlab/"
rsync -a "${EXCLUDES[@]}" neural_ode_v2/scripts/ "$BUCK/scripts/"
rsync -a "${EXCLUDES[@]}" neural_ode_v2/data/ "$BUCK/data/"
rsync -a "${EXCLUDES[@]}" training/artifacts/ "$BUCK/artifacts/"
cp train.log "$BUCK/artifacts/train.log"
cp ROUTE.md "$BUCK/docs/ROUTE.md"
cp requirements-alignment.md "$BUCK/docs/requirements-alignment.md"
cp neural_ode_v2/CHANGE_LOG.md "$BUCK/docs/CHANGE_LOG.md"
```

Expected: about 225 MB of effective copied data and artifacts; generated caches are absent.

- [ ] **Step 3: Verify byte-for-byte data and artifact copies**

Run:

```bash
set -e
MIGRATION=dc_dc_converter_neural_ode/.migration
BUCK=dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck

(cd neural_ode_v2/data && find . -type f \
  ! -name '.DS_Store' ! -name '*.pyc' ! -name '*.slxc' \
  ! -path '*/__pycache__/*' ! -path '*/slprj/*' -print0 \
  | sort -z | xargs -0 shasum -a 256) > "$MIGRATION/source-data.sha256"
(cd "$BUCK/data" && find . -type f \
  ! -name '.DS_Store' ! -name '*.pyc' ! -name '*.slxc' \
  ! -path '*/__pycache__/*' ! -path '*/slprj/*' -print0 \
  | sort -z | xargs -0 shasum -a 256) > "$MIGRATION/copied-data.sha256"
diff -u "$MIGRATION/source-data.sha256" "$MIGRATION/copied-data.sha256"

(cd training/artifacts && find . -type f ! -name '.DS_Store' -print0 \
  | sort -z | xargs -0 shasum -a 256) > "$MIGRATION/source-artifacts.sha256"
(cd "$BUCK/artifacts" && find . -type f ! -name '.DS_Store' ! -name 'train.log' -print0 \
  | sort -z | xargs -0 shasum -a 256) > "$MIGRATION/copied-artifacts.sha256"
diff -u "$MIGRATION/source-artifacts.sha256" "$MIGRATION/copied-artifacts.sha256"

shasum -a 256 -c "$MIGRATION/legacy-content.sha256" >/dev/null
```

Expected: both `diff` commands and the legacy checksum check exit 0.

- [ ] **Step 4: Remove known trailing whitespace from the new copies only**

The legacy `training/data.py` and `project.json` contain two whitespace-only formatting defects. Normalize trailing whitespace in the new copies so repository checks pass, without editing legacy files:

```bash
python3 - <<'PY'
from pathlib import Path

files = [
    Path("dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/training/data.py"),
    Path("dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/configs/known_pwm_matlab_v2/project.json"),
]
for path in files:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
PY
```

Expected: only trailing spaces change in the new copies; Python behavior and parsed JSON remain identical.

- [ ] **Step 5: Commit copied effective source and resources**

```bash
git add \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/requirements.txt \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/training \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/analysis \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/configs \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/matlab \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/scripts \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/docs \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/tests/__init__.py
git diff --cached --check
git commit -m "chore: copy Buck project into topology workspace"
```

Expected: data and artifacts remain ignored; commit contains copied code, config, Simulink model, scripts, and historical docs only.

---

### Task 3: Adapt the Buck training package and entry paths

**Files:**
- Create: `.../buck/tests/test_project_paths.py`
- Modify: `.../buck/train.py`
- Modify: `.../buck/training/loops.py`
- Modify: `.../buck/training/evaluate.py`

- [ ] **Step 1: Write the failing entry/path test**

Create `dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/tests/test_project_paths.py`:

```python
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BUCK_ROOT = Path(__file__).resolve().parents[1]
ENTRY = BUCK_ROOT / "train.py"


def load_entry():
    sys.path.insert(0, str(BUCK_ROOT))
    try:
        spec = importlib.util.spec_from_file_location("new_buck_entry", ENTRY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class ProjectPathTests(unittest.TestCase):
    def test_entry_defaults_are_inside_buck_project(self):
        entry = load_entry()
        self.assertEqual(
            Path(entry.DEFAULT_CONFIG).resolve(),
            BUCK_ROOT / "configs/known_pwm_matlab_v2/project.json",
        )
        self.assertEqual(
            Path(entry.DEFAULT_DATA_DIR).resolve(),
            BUCK_ROOT / "data/processed/known_pwm_matlab_v2",
        )
        self.assertEqual(
            Path(entry.DEFAULT_OUT_DIR).resolve(), BUCK_ROOT / "artifacts"
        )

    def test_help_is_cwd_independent(self):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, str(ENTRY), "--help"],
                cwd=cwd,
                text=True,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Buck 两模式 Neural Event ODE", result.stdout)

    def test_runtime_python_has_no_legacy_path_reference(self):
        files = [ENTRY, *sorted((BUCK_ROOT / "training").glob("*.py"))]
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("Buck_neural_ode_v2", text, path)
            self.assertNotIn("neural_ode_v2", text, path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails against the raw copy**

Run:

```bash
python3 -m unittest \
  dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.tests.test_project_paths -v
```

Expected: FAIL/ERROR because the copied entry still imports top-level `data`, `model`, `train`, `evaluate`, or `checkpoint`, and still points to `Buck_neural_ode_v2`.

- [ ] **Step 3: Adapt the main entry imports and defaults**

In `buck/train.py`, replace legacy imports and path constants with:

```python
from pathlib import Path

import torch

from training.data import load_data, normalize, build_pairs
from training.model import TwoModeVectorField
from training.loops import run_local_mode, run_stage
from training.evaluate import (
    evaluate_all,
    evaluate_by_voltage,
    print_report,
    print_by_voltage,
    save_plot,
    save_voltage_plots,
)
from training.checkpoint import load_checkpoint

BUCK_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = str(BUCK_ROOT / "configs" / "known_pwm_matlab_v2" / "project.json")
DEFAULT_DATA_DIR = str(BUCK_ROOT / "data" / "processed" / "known_pwm_matlab_v2")
DEFAULT_OUT_DIR = str(BUCK_ROOT / "artifacts")
```

Change only the output argument default:

```python
ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
```

Update the module docstring example to invoke `python3 <buck>/train.py` without a legacy `PYTHONPATH` requirement. Do not change any CLI option, model, training, or evaluation logic.

- [ ] **Step 4: Convert copied internal imports to package-relative imports**

At the top of `buck/training/loops.py`, use:

```python
from .data import make_windows, collate
from .losses import mse_loss, ripple_loss
from .model import integrate, tau_grid
from .checkpoint import save_checkpoint
```

At the top of `buck/training/evaluate.py`, use:

```python
from .model import integrate, tau_grid
```

No other runtime logic changes are permitted in these files.

- [ ] **Step 5: Run the entry/path test and existing syntax checks**

Run:

```bash
python3 -m unittest \
  dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.tests.test_project_paths -v
python3 -m compileall -q \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/training
```

Expected: 3 tests PASS; compileall exits 0.

- [ ] **Step 6: Commit the independent Buck training entry**

```bash
git add \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/training \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/tests/test_project_paths.py
git diff --cached --check
git commit -m "refactor: adapt Buck training to topology-local paths"
```

---

### Task 4: Adapt post-training analysis tools

**Files:**
- Create: `.../buck/analysis/_bootstrap.py`
- Create: `.../buck/tests/test_analysis_paths.py`
- Modify: all four `.../buck/analysis/plot_*.py`

- [ ] **Step 1: Write the failing analysis path test**

Create `buck/tests/test_analysis_paths.py`:

```python
import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BUCK_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = BUCK_ROOT / "analysis"


class AnalysisPathTests(unittest.TestCase):
    def test_analysis_modules_import_from_any_cwd(self):
        sys.path.insert(0, str(ANALYSIS_DIR))
        try:
            for name in (
                "plot_waveforms",
                "plot_extra_hi",
                "plot_ripple",
                "plot_ripple_vc",
            ):
                module = importlib.import_module(name)
                self.assertIsNotNone(module)
        finally:
            sys.path.pop(0)

    def test_bootstrap_defaults_are_local(self):
        sys.path.insert(0, str(ANALYSIS_DIR))
        try:
            bootstrap = importlib.import_module("_bootstrap")
        finally:
            sys.path.pop(0)
        self.assertEqual(Path(bootstrap.DEFAULT_CONFIG), BUCK_ROOT / "configs/known_pwm_matlab_v2/project.json")
        self.assertEqual(Path(bootstrap.DEFAULT_DATA_DIR), BUCK_ROOT / "data/processed/known_pwm_matlab_v2")
        self.assertEqual(Path(bootstrap.DEFAULT_CKPT), BUCK_ROOT / "artifacts/checkpoint.pt")
        self.assertEqual(Path(bootstrap.DEFAULT_WAVEFORM_DIR), BUCK_ROOT / "artifacts/waveforms")

    def test_argument_scripts_have_cwd_independent_help(self):
        with tempfile.TemporaryDirectory() as cwd:
            for script in ("plot_waveforms.py", "plot_extra_hi.py"):
                result = subprocess.run(
                    [sys.executable, str(ANALYSIS_DIR / script), "--help"],
                    cwd=cwd,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_analysis_python_has_no_legacy_path_reference(self):
        for path in ANALYSIS_DIR.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("Buck_neural_ode_v2", text, path)
            self.assertNotIn("neural_ode_v2", text, path)
            self.assertNotIn("training/artifacts", text, path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it and verify the copied scripts fail**

Run:

```bash
python3 -m unittest \
  dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.tests.test_analysis_paths -v
```

Expected: FAIL/ERROR because `_bootstrap.py` does not exist and copied scripts import legacy top-level modules and paths.

- [ ] **Step 3: Create one analysis bootstrap for imports and paths**

Create `buck/analysis/_bootstrap.py`:

```python
"""CWD-independent imports and default paths for Buck analysis scripts."""

from pathlib import Path
import sys

BUCK_ROOT = Path(__file__).resolve().parents[1]
if str(BUCK_ROOT) not in sys.path:
    sys.path.insert(0, str(BUCK_ROOT))

DEFAULT_CONFIG = str(BUCK_ROOT / "configs" / "known_pwm_matlab_v2" / "project.json")
DEFAULT_DATA_DIR = str(BUCK_ROOT / "data" / "processed" / "known_pwm_matlab_v2")
DEFAULT_CKPT = str(BUCK_ROOT / "artifacts" / "checkpoint.pt")
DEFAULT_WEIGHTS = str(BUCK_ROOT / "artifacts" / "best.pt")
DEFAULT_ARTIFACTS_DIR = str(BUCK_ROOT / "artifacts")
DEFAULT_WAVEFORM_DIR = str(BUCK_ROOT / "artifacts" / "waveforms")
```

- [ ] **Step 4: Adapt imports and output defaults in all analysis scripts**

Use these imports in `plot_waveforms.py`:

```python
from _bootstrap import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DEFAULT_CKPT, DEFAULT_WAVEFORM_DIR
from training.model import TwoModeVectorField, integrate, tau_grid
from training.data import load_data, normalize
from training.checkpoint import load_checkpoint
```

Set its `--out-dir` default to `DEFAULT_WAVEFORM_DIR` and remove copied `SCRIPT_DIR`/legacy default definitions.

Use these imports in `plot_extra_hi.py`:

```python
from _bootstrap import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_WEIGHTS,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_WAVEFORM_DIR,
)
from training.data import load_data, normalize
from training.model import TwoModeVectorField, integrate, tau_grid
```

Set `--out-dir` to `DEFAULT_WAVEFORM_DIR`, and build `metrics_eval.json` from `DEFAULT_ARTIFACTS_DIR`.

Use these imports in both `plot_ripple.py` and `plot_ripple_vc.py`:

```python
from _bootstrap import DEFAULT_WAVEFORM_DIR
from plot_waveforms import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_CKPT,
    SPLIT_TITLES,
    load_model,
    rollout,
    per_cycle_metrics,
    denorm,
)
from training.data import load_data, normalize
```

Set each `OUT` to `os.path.join(DEFAULT_WAVEFORM_DIR, "ripple_overview.png")` or `os.path.join(DEFAULT_WAVEFORM_DIR, "vc_ripple_overview.png")`. Update docstrings to show new Buck-local output paths.

- [ ] **Step 5: Run analysis tests and compile checks**

```bash
python3 -m unittest \
  dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.tests.test_analysis_paths -v
python3 -m compileall -q \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/analysis
```

Expected: 4 tests PASS; compileall exits 0.

- [ ] **Step 6: Commit analysis adaptation**

```bash
git add \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/analysis \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/tests/test_analysis_paths.py
git diff --cached --check
git commit -m "refactor: make Buck analysis topology-local"
```

---

### Task 5: Lock data-generation paths and dependencies

**Files:**
- Create: `.../buck/tests/test_generation_layout.py`
- Modify: `.../buck/requirements.txt`
- Verify without modification unless a test proves otherwise: `.../buck/matlab/scripts/generate_dataset.m`
- Verify without modification unless a test proves otherwise: `.../buck/scripts/mat_to_npz.py`

- [ ] **Step 1: Write the failing generation-layout test**

Create `buck/tests/test_generation_layout.py`:

```python
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BUCK_ROOT = Path(__file__).resolve().parents[1]


class GenerationLayoutTests(unittest.TestCase):
    def test_required_generation_files_exist(self):
        required = [
            BUCK_ROOT / "configs/known_pwm_matlab_v2/project.json",
            BUCK_ROOT / "matlab/scripts/buck_ideal_ccm.slx",
            BUCK_ROOT / "matlab/scripts/generate_dataset.m",
            BUCK_ROOT / "scripts/mat_to_npz.py",
        ]
        for path in required:
            self.assertTrue(path.is_file(), path)

    def test_matlab_two_parent_rule_resolves_buck_root(self):
        script_dir = BUCK_ROOT / "matlab/scripts"
        self.assertEqual(script_dir.parents[1], BUCK_ROOT)
        text = (script_dir / "generate_dataset.m").read_text(encoding="utf-8")
        self.assertIn('project_root = fileparts(fileparts(script_dir));', text)
        self.assertNotIn("Buck_neural_ode_v2", text)
        self.assertNotIn("neural_ode_v2", text)

    def test_converter_help_is_cwd_independent(self):
        script = BUCK_ROOT / "scripts/mat_to_npz.py"
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=cwd,
                text=True,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--raw-dir", result.stdout)
        self.assertIn("--out-dir", result.stdout)

    def test_converter_dependency_is_declared(self):
        requirements = (BUCK_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("scipy>=", requirements.lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it and verify the undeclared SciPy dependency fails**

```bash
python3 -m unittest \
  dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.tests.test_generation_layout -v
```

Expected: three tests PASS and `test_converter_dependency_is_declared` FAIL because copied `requirements.txt` does not list SciPy.

- [ ] **Step 3: Declare the converter dependency without changing model dependencies**

Append to `buck/requirements.txt`:

```text
scipy>=1.10
```

Do not refactor `generate_dataset.m` or `mat_to_npz.py`: their existing script-relative two-parent path logic already resolves the new Buck root and is covered by the test.

- [ ] **Step 4: Re-run generation tests**

```bash
python3 -m unittest \
  dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.tests.test_generation_layout -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit generation layout validation**

```bash
git add \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/requirements.txt \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/tests/test_generation_layout.py
git diff --cached --check
git commit -m "test: lock Buck data-generation layout"
```

---

### Task 6: Prove copied data and model behavior are equivalent

**Files:**
- Create: `.../buck/tests/test_regression_equivalence.py`

- [ ] **Step 1: Add deterministic data and rollout regression tests**

Create `buck/tests/test_regression_equivalence.py`:

```python
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

BUCK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
LEGACY_TRAINING = REPO_ROOT / "training"
LEGACY_DATA_FILE = REPO_ROOT / "neural_ode_v2/data/processed/known_pwm_matlab_v2/train.npz"
NEW_DATA_FILE = BUCK_ROOT / "data/processed/known_pwm_matlab_v2/train.npz"
CONFIG = BUCK_ROOT / "configs/known_pwm_matlab_v2/project.json"
BEST = BUCK_ROOT / "artifacts/best.pt"


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy_data = load_file_module("legacy_buck_data", LEGACY_TRAINING / "data.py")
legacy_model = load_file_module("legacy_buck_model", LEGACY_TRAINING / "model.py")
sys.path.insert(0, str(BUCK_ROOT))
from training import data as new_data
from training import model as new_model
sys.path.pop(0)


class RegressionEquivalenceTests(unittest.TestCase):
    def test_copied_train_npz_is_byte_identical(self):
        self.assertEqual(LEGACY_DATA_FILE.read_bytes(), NEW_DATA_FILE.read_bytes())

    def test_loaded_normalized_data_and_windows_are_identical(self):
        refs = json.loads(CONFIG.read_text(encoding="utf-8"))["normalization"]
        with tempfile.TemporaryDirectory() as old_dir, tempfile.TemporaryDirectory() as new_dir:
            os.symlink(LEGACY_DATA_FILE, Path(old_dir) / "train.npz")
            os.symlink(NEW_DATA_FILE, Path(new_dir) / "train.npz")
            old = legacy_data.normalize(legacy_data.load_data(old_dir), refs)["train"]
            new = new_data.normalize(new_data.load_data(new_dir), refs)["train"]
        self.assertTrue(torch.equal(old["states_norm"], new["states_norm"]))
        self.assertTrue(torch.equal(old["vs_norm"], new["vs_norm"]))
        self.assertTrue(torch.equal(old["mode"], new["mode"]))
        self.assertEqual(
            legacy_data.make_windows(old, 10),
            new_data.make_windows(new, 10),
        )

    def test_existing_weights_produce_identical_one_cycle_rollout(self):
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        refs = cfg["normalization"]
        with tempfile.TemporaryDirectory() as data_dir:
            os.symlink(NEW_DATA_FILE, Path(data_dir) / "train.npz")
            data = new_data.normalize(new_data.load_data(data_dir), refs)["train"]
        state = torch.load(BEST, map_location="cpu", weights_only=True)
        old = legacy_model.TwoModeVectorField(cfg["model"], cfg["circuit"]["duty_cycle"]).double()
        new = new_model.TwoModeVectorField(cfg["model"], cfg["circuit"]["duty_cycle"]).double()
        old.load_state_dict(state)
        new.load_state_dict(state)
        y0 = data["states_norm"][:1, 0, :]
        vs = data["vs_norm"][:1].unsqueeze(1)
        old.set_vs(vs)
        new.set_vs(vs)
        tau_old = legacy_model.tau_grid(1, torch.float64, "cpu")
        tau_new = new_model.tau_grid(1, torch.float64, "cpu")
        with torch.no_grad():
            pred_old = legacy_model.integrate(old, y0, tau_old, "standard", torch.float64, "cpu")
            pred_new = new_model.integrate(new, y0, tau_new, "standard", torch.float64, "cpu")
        torch.testing.assert_close(pred_old, pred_new, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run regression equivalence tests**

```bash
python3 -m unittest \
  dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.tests.test_regression_equivalence -v
```

Expected: 3 tests PASS. If any comparison fails, stop and diagnose; do not loosen tolerances because copied code and files should be exactly equivalent.

- [ ] **Step 3: Commit regression tests**

```bash
git add \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/tests/test_regression_equivalence.py
git diff --cached --check
git commit -m "test: verify Buck migration equivalence"
```

---

### Task 7: Document the workspace and self-contained Buck project

**Files:**
- Create: `dc_dc_converter_neural_ode/README.md`
- Create: `dc_dc_converter_neural_ode/AGENTS.md`
- Create: `dc_dc_converter_neural_ode/docs/architecture.md`
- Create: `dc_dc_converter_neural_ode/docs/topology_catalog.md`
- Create: `.../buck/README.md`
- Create: `.../buck/tests/test_documentation.py`
- Modify: `.../buck/docs/{ROUTE.md,requirements-alignment.md,CHANGE_LOG.md}` only by prepending a history notice

- [ ] **Step 1: Write the failing documentation test**

Create `buck/tests/test_documentation.py`:

```python
import unittest
from pathlib import Path

BUCK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[4]


class DocumentationTests(unittest.TestCase):
    def test_workspace_documents_exist(self):
        for relative in ("README.md", "AGENTS.md", "docs/architecture.md", "docs/topology_catalog.md"):
            self.assertTrue((WORKSPACE / relative).is_file(), relative)

    def test_catalog_marks_only_buck_implemented(self):
        catalog = (WORKSPACE / "docs/topology_catalog.md").read_text(encoding="utf-8")
        self.assertIn("Buck | 已实现", catalog)
        for name in ("Boost", "Ćuk", "Forward", "Flyback", "DAB"):
            self.assertIn(f"{name} | 规划", catalog)

    def test_buck_readme_uses_new_entry(self):
        text = (BUCK_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("topologies/non_isolated_unidirectional/buck/train.py", text)
        self.assertNotIn("python3 training/train_buck.py", text)

    def test_workspace_rules_protect_legacy_project(self):
        text = (WORKSPACE / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("不得修改旧项目", text)
        self.assertIn("拓扑自包含", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it and verify documentation is missing**

```bash
python3 -m unittest \
  dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.tests.test_documentation -v
```

Expected: FAIL because workspace-level docs and Buck README do not exist yet.

- [ ] **Step 3: Create workspace README and rules**

Create `dc_dc_converter_neural_ode/README.md` with:

````markdown
# DC–DC Converter Neural ODE

本目录是 DC–DC 变换器 Neural ODE 的统一工作区。每个拓扑是一个自包含项目，拥有独立的数据生成、配置、训练、评估和产物目录。

## 已实现

- Buck：`topologies/non_isolated_unidirectional/buck/`

其他拓扑均处于规划状态，见 `docs/topology_catalog.md`。

## Buck 快速开始

```bash
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py --smoke
```

现有仓库根目录中的旧 Buck 项目保留不动；后续开发以本工作区为准。
````

Create `dc_dc_converter_neural_ode/AGENTS.md` with:

```markdown
# DC–DC 工作区约束

- 拓扑自包含：每个拓扑拥有自己的训练入口、配置、仿真、数据、模型、评估和产物。
- 不得修改旧项目：`../training/` 与 `../neural_ode_v2/` 仅作为迁移来源和回归基线。
- 未实现拓扑不得创建空实现，也不得在文档中标记为已支持。
- 只有两个及以上真实拓扑出现稳定重复后，才允许提取 `shared/`。
- 新增拓扑必须放入 `topologies/` 下对应分类，并先编写拓扑级 README 和验收标准。
- 数据、训练产物和 Simulink 缓存不得提交 Git。
```

- [ ] **Step 4: Create architecture and topology catalog docs**

Create `docs/architecture.md` describing these exact dependency rules:

```markdown
# 架构

## 单元边界

每个拓扑目录都是独立可运行单元。拓扑可以拥有不同的状态维数、离散模式、网络结构、积分策略、损失和评估指标。

## 当前依赖方向

`拓扑入口 → 本拓扑 training/、configs/、data/、artifacts/`

禁止新工作区代码运行时引用旧项目，禁止拓扑横向导入其他拓扑。当前不设统一 framework、registry 或总 CLI。

## 公共代码演化

接入第二个真实拓扑后，先保持自包含；仅将两个实现中已经相同且稳定的部分提取到未来的 `shared/`。模式建模接口在具体拓扑需求明确后另行设计。
```

Create `docs/topology_catalog.md` with a table containing these rows:

```markdown
# 拓扑目录

| 大类 | 拓扑 | 状态 | 规划路径 |
| --- | --- | --- | --- |
| 非隔离单向 | Buck | 已实现 | `non_isolated_unidirectional/buck/` |
| 非隔离单向 | Boost | 规划 | `non_isolated_unidirectional/boost/` |
| 非隔离单向 | Buck–Boost | 规划 | `non_isolated_unidirectional/buck_boost/` |
| 非隔离单向 | Ćuk | 规划 | `non_isolated_unidirectional/cuk/` |
| 多相扩展 | 两相交错 Buck | 规划 | `multiphase/two_phase_interleaved_buck/` |
| 隔离单向 | Forward | 规划 | `isolated_unidirectional/forward/` |
| 隔离单向 | Flyback | 规划 | `isolated_unidirectional/flyback/` |
| 隔离单向 | Push-Pull | 规划 | `isolated_unidirectional/push_pull/` |
| 隔离单向 | Half-Bridge | 规划 | `isolated_unidirectional/half_bridge/` |
| 隔离单向 | Full-Bridge | 规划 | `isolated_unidirectional/full_bridge/` |
| 双向 | 非隔离双向 Buck/Boost | 规划 | `bidirectional/non_isolated_buck_boost/` |
| 双向 | DAB | 规划 | `bidirectional/dab/` |
```

Do not create any planned topology directory.

- [ ] **Step 5: Create the operational Buck README**

Create `buck/README.md` with exactly this operational content:

````markdown
# Buck Neural Event ODE

分类：非隔离单向变换器 → 基本单电感型 → Buck。

本目录是可独立运行的 Buck Neural Event ODE 项目，包含数据生成、数据转换、训练、评估、分析、数据和训练产物。后续修改以本目录为准，不会同步到仓库旧 Buck 项目。

## 固定物理与数值约束

- CCM；状态为 `(iL, vC)`，外部输入为 `vS`。
- D=0.5，20 kHz，每周期 100 个采样区间。
- ON/OFF 使用两个独立 MLP。
- RK4 固定步长 `h=0.01`（归一化周期时间）。
- 当前默认反传路线为 `standard`。

## 数据链

`matlab/scripts/buck_ideal_ccm.slx` → `generate_dataset.m` → `data/raw/` → `scripts/mat_to_npz.py` → `data/processed/` → `train.py`

## 目录

- `configs/`：唯一事实源配置。
- `training/`：数据、模型、训练循环、损失、评估和 checkpoint。
- `analysis/`：训练后波形与纹波诊断。
- `matlab/`、`scripts/`：数据生成与转换。
- `data/`、`artifacts/`：完整本地数据和训练产物，不提交 Git。

## 命令

以下命令从仓库根目录执行：

```bash
# Full training
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py

# Smoke test
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py --smoke

# Resume copied checkpoint
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py \
  --resume dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/checkpoint.pt

# Evaluate copied best weights
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py \
  --evaluate-only \
  --resume dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/best.pt
```

## 历史与需求

- [反传路线](docs/ROUTE.md)
- [需求对齐](docs/requirements-alignment.md)
- [数据和配置变更](docs/CHANGE_LOG.md)
````

- [ ] **Step 6: Mark copied design records as historical context**

Prepend this block to each of `buck/docs/ROUTE.md`, `buck/docs/requirements-alignment.md`, and `buck/docs/CHANGE_LOG.md`:

```markdown
> [!NOTE]
> 本文件是旧 Buck 项目的历史记录副本。正文中的旧路径保留当时语境；当前可执行路径和命令以 `../README.md` 为准。

```

Do not rewrite their historical technical decisions.

- [ ] **Step 7: Run documentation tests**

```bash
python3 -m unittest \
  dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.tests.test_documentation -v
```

Expected: 4 tests PASS.

- [ ] **Step 8: Commit documentation**

```bash
git add \
  dc_dc_converter_neural_ode/README.md \
  dc_dc_converter_neural_ode/AGENTS.md \
  dc_dc_converter_neural_ode/docs/architecture.md \
  dc_dc_converter_neural_ode/docs/topology_catalog.md \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/README.md \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/docs \
  dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/tests/test_documentation.py
git diff --cached --check
git commit -m "docs: document self-contained DC-DC topology workspace"
```

---

### Task 8: Run end-to-end smoke checks and prove legacy preservation

**Files:**
- No tracked production file changes
- Create at runtime, ignored: `dc_dc_converter_neural_ode/.migration/legacy-paths.after`
- Temporary only: one mini dataset, one modified checkpoint copy, and smoke outputs under `mktemp -d`

- [ ] **Step 1: Run the complete Buck unit/regression suite**

```bash
python3 -m unittest discover \
  -s dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/tests \
  -p 'test_*.py' -v
```

Expected: 18 tests PASS (3 project path + 4 analysis + 4 generation + 3 regression + 4 documentation).

- [ ] **Step 2: Create a temporary 20-cycle mini dataset and resumable checkpoint copy**

Run:

```bash
set -e
BUCK=dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck
TMP=$(mktemp -d)
printf '%s' "$TMP" > /tmp/dcdc-smoke-path
mkdir -p "$TMP/data" "$TMP/out" "$TMP/eval"

python3 - "$BUCK" "$TMP" <<'PY'
from pathlib import Path
import sys
import numpy as np
import torch

buck = Path(sys.argv[1])
tmp = Path(sys.argv[2])
source = buck / "data/processed/known_pwm_matlab_v2"
for split in ("train", "validation"):
    d = np.load(source / f"{split}.npz", allow_pickle=True)
    np.savez(
        tmp / "data" / f"{split}.npz",
        states=d["states"][:1, :2001, :],
        input_voltage=d["input_voltage"][:1],
        interval_mode=d["interval_mode"][:2000],
    )

checkpoint = torch.load(buck / "artifacts/checkpoint.pt", map_location="cpu", weights_only=True)
checkpoint["stage"] = "twenty_cycle"
checkpoint["epoch"] = 59
checkpoint["completed_stages"] = [
    name for name in checkpoint["completed_stages"] if name != "twenty_cycle"
]
torch.save(checkpoint, tmp / "resume-smoke.pt")
PY
```

Expected: temporary train/validation NPZs each contain one 20-cycle trajectory; the temporary checkpoint requests exactly one final twenty-cycle epoch. `/tmp/dcdc-smoke-path` contains the temporary directory path for the remaining smoke steps.

- [ ] **Step 3: Run one real resumed twenty-cycle optimization epoch against temporary outputs**

Reload the temporary paths, then run:

```bash
BUCK=dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck
TMP=$(cat /tmp/dcdc-smoke-path)
python3 "$BUCK/train.py" \
  --smoke \
  --device cpu \
  --data-dir "$TMP/data" \
  --out-dir "$TMP/out" \
  --resume "$TMP/resume-smoke.pt" \
  --backprop standard \
  --max-window-cycles 20 \
  --max-windows 1 \
  --epochs 60 \
  --checkpoint-every 1
```

Expected: `twenty_cycle` runs epoch `60/60`, then 1/10-cycle train/validation evaluation completes; `$TMP/out/checkpoint.pt`, `best.pt`, `metrics.json`, and `steady_state_compare.png` exist.

- [ ] **Step 4: Run independent evaluate-only smoke from copied best weights**

```bash
BUCK=dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck
TMP=$(cat /tmp/dcdc-smoke-path)
python3 "$BUCK/train.py" \
  --smoke \
  --device cpu \
  --data-dir "$TMP/data" \
  --out-dir "$TMP/eval" \
  --evaluate-only \
  --resume "$BUCK/artifacts/best.pt" \
  --backprop standard
```

Expected: exit code 0; `$TMP/eval/metrics_eval.json` and `$TMP/eval/steady_state_compare_eval.png` exist.

- [ ] **Step 5: Verify no runtime source references old project paths**

Run:

```bash
if grep -R -n -E 'Buck_neural_ode_v2|neural_ode_v2|python3 training/train_buck.py' \
  "$BUCK/train.py" "$BUCK/training" "$BUCK/analysis" "$BUCK/scripts" "$BUCK/matlab"; then
  echo 'legacy runtime reference found' >&2
  exit 1
fi
```

Expected: grep returns no matches.

- [ ] **Step 6: Recheck legacy hashes and pre-existing Git status**

Run:

```bash
set -e
MIGRATION=dc_dc_converter_neural_ode/.migration

{
  find training neural_ode_v2 -type f \
    ! -name '.DS_Store' ! -name '*.pyc' ! -name '*.slxc' \
    ! -path '*/__pycache__/*' ! -path '*/slprj/*' -print0
  printf '%s\0' AGENTS.md ROUTE.md requirements-alignment.md train.log
} | LC_ALL=C sort -z | tr '\0' '\n' > "$MIGRATION/legacy-paths.after"

diff -u "$MIGRATION/legacy-paths.before" "$MIGRATION/legacy-paths.after"
shasum -a 256 -c "$MIGRATION/legacy-content.sha256" >/dev/null

git status --porcelain=v1 --untracked-files=all \
  | grep -v '^?? dc_dc_converter_neural_ode/' \
  > "$MIGRATION/git-status.after"
diff -u "$MIGRATION/git-status.before" "$MIGRATION/git-status.after"
```

Expected: path-inventory diff, checksum validation, and Git-status diff all exit 0. This detects added, deleted, or modified legacy files and proves all pre-existing files and user changes stayed untouched.

- [ ] **Step 7: Remove temporary smoke data and inspect final repository status**

```bash
TMP=$(cat /tmp/dcdc-smoke-path)
rm -rf "$TMP"
rm -f /tmp/dcdc-smoke-path

git status --short
git log --oneline -8
```

Expected: no uncommitted tracked changes under `dc_dc_converter_neural_ode/`; copied `data/`, `artifacts/`, `.migration/`, and caches remain ignored; the pre-existing root worktree changes remain exactly as captured.

---

## Final acceptance checklist

- [ ] Only `dc_dc_converter_neural_ode/` was added or changed by implementation.
- [ ] Complete effective Buck data and artifacts were copied, not linked.
- [ ] New Python/MATLAB/converter defaults resolve only within the new Buck root.
- [ ] Existing weights load and produce bit-identical one-cycle rollout.
- [ ] A real checkpoint-resume optimization epoch succeeds from the new entry.
- [ ] Evaluate-only succeeds from copied `best.pt`.
- [ ] All 18 tests pass.
- [ ] No unimplemented topology directory exists.
- [ ] No `framework/`, registry, unified CLI, or rectifier component exists.
- [ ] Legacy hashes and pre-existing Git status are unchanged.
