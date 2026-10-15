# Project-Local MATLAB Agentic Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install and verify the official MATLAB Agentic Toolkit and MATLAB MCP Server for this repository without modifying user-global configuration.

**Architecture:** Keep downloaded artifacts under the ignored project-local `.tools/` directory. Pi discovers only the Toolkit's required `matlab-core` skills through `.pi/settings.json`, while the existing project `.mcp.json` starts the local MATLAB MCP binary against the mounted R2025a installation.

**Tech Stack:** MATLAB R2025a, MATLAB Agentic Toolkit `2026.08.13`, MATLAB MCP Server `v0.12.0`, Pi `0.84.3`, pi-mcp-adapter, JSON, Git.

---

## File Map

- Modify: `.gitignore` — exclude downloaded Toolkit/MCP artifacts and local MCP logs.
- Modify: `.mcp.json` — retain `atomicviz` and register the project-local `matlab` stdio server.
- Create: `.pi/settings.json` — add the project-local `matlab-core` skill-group path.
- Create locally, ignored: `.tools/matlab-agentic-toolkit/` — official Toolkit checkout pinned to tag `2026.08.13`.
- Create locally, ignored: `.tools/matlab-mcp-server/matlab-mcp-server` — official MCP Server `v0.12.0` Apple Silicon binary.

### Task 1: Establish Project-Local Boundaries

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Capture the relevant user-global configuration hashes**

Run:

```bash
mkdir -p /tmp/matlab-agentic-toolkit-install
for file in \
  "$HOME/.pi/agent/settings.json" \
  "$HOME/.codex/config.toml" \
  "$HOME/.claude/settings.json"; do
  if [ -f "$file" ]; then
    shasum -a 256 "$file"
  fi
done > /tmp/matlab-agentic-toolkit-install/global-config.before.sha256
```

Expected: The baseline file contains hashes and no project file changes occur.

- [ ] **Step 2: Verify the project-local artifact paths are not ignored yet**

Run:

```bash
git check-ignore -q .tools/matlab-mcp-server/matlab-mcp-server
```

Expected: Exit status `1` before the ignore rule is added.

- [ ] **Step 3: Add the project-local artifact exclusions**

Append exactly this section to `.gitignore`:

```gitignore

# Project-local agent and MCP downloads
.tools/
.pi/mcp-logs/
```

- [ ] **Step 4: Verify ignore behavior**

Run:

```bash
git check-ignore -v .tools/matlab-mcp-server/matlab-mcp-server
```

Expected: Output identifies `.gitignore` and the `.tools/` rule.

- [ ] **Step 5: Commit the boundary rule**

```bash
git add .gitignore
git commit -m "chore: ignore project-local MATLAB agent downloads"
```

### Task 2: Install the Official Toolkit Locally

**Files:**
- Create locally, ignored: `.tools/matlab-agentic-toolkit/`

- [ ] **Step 1: Clone the pinned official Toolkit release**

Run:

```bash
mkdir -p .tools
git clone --depth 1 --branch 2026.08.13 \
  https://github.com/matlab/matlab-agentic-toolkit.git \
  .tools/matlab-agentic-toolkit
```

Expected: Git checks out tag `2026.08.13` under the project.

- [ ] **Step 2: Verify provenance and required skills**

Run:

```bash
git -C .tools/matlab-agentic-toolkit remote get-url origin
git -C .tools/matlab-agentic-toolkit describe --tags --exact-match
test -d .tools/matlab-agentic-toolkit/skills-catalog/matlab-core
find .tools/matlab-agentic-toolkit/skills-catalog/matlab-core \
  -name SKILL.md -type f | sort
```

Expected: Origin is `https://github.com/matlab/matlab-agentic-toolkit.git`, the exact tag is `2026.08.13`, and at least one `SKILL.md` is listed.

### Task 3: Install the Official MCP Binary Locally

**Files:**
- Create locally, ignored: `.tools/matlab-mcp-server/matlab-mcp-server`

- [ ] **Step 1: Download the pinned Apple Silicon release**

Run:

```bash
mkdir -p .tools/matlab-mcp-server
curl -fL \
  https://github.com/matlab/matlab-mcp-server/releases/download/v0.12.0/matlab-mcp-server-macos-arm64 \
  -o .tools/matlab-mcp-server/matlab-mcp-server
chmod 755 .tools/matlab-mcp-server/matlab-mcp-server
```

Expected: `curl` exits `0` and creates an executable approximately 18 MB in size.

- [ ] **Step 2: Remove quarantine only from the local binary if present**

Run:

```bash
if xattr -p com.apple.quarantine .tools/matlab-mcp-server/matlab-mcp-server >/dev/null 2>&1; then
  xattr -d com.apple.quarantine .tools/matlab-mcp-server/matlab-mcp-server
fi
```

Expected: The command exits `0`; no file outside this repository is touched.

- [ ] **Step 3: Verify architecture and version**

Run:

```bash
file .tools/matlab-mcp-server/matlab-mcp-server
.tools/matlab-mcp-server/matlab-mcp-server --version
```

Expected: `file` reports a Mach-O ARM64 executable and the server reports version `0.12.0`.

### Task 4: Configure Project-Local Pi Skills and MCP

**Files:**
- Create: `.pi/settings.json`
- Modify: `.mcp.json`

- [ ] **Step 1: Demonstrate that project Pi skill configuration is absent**

Run:

```bash
test -f .pi/settings.json
```

Expected: Exit status `1`.

- [ ] **Step 2: Create project-local Pi skill discovery settings**

Create `.pi/settings.json` with exactly:

```json
{
  "skills": [
    "../.tools/matlab-agentic-toolkit/skills-catalog/matlab-core"
  ]
}
```

- [ ] **Step 3: Add MATLAB without changing the existing AtomicViz entry**

Set `.mcp.json` to:

```json
{
  "mcpServers": {
    "atomicviz": {
      "type": "http",
      "url": "http://127.0.0.1:3939/mcp"
    },
    "matlab": {
      "command": "./.tools/matlab-mcp-server/matlab-mcp-server",
      "args": [
        "--matlab-root=/Volumes/MacApp/MATLAB_R2025a.app",
        "--initial-working-folder=/Users/antepil/Documents/ChatGPT/dc_dc_converter_neural_ode",
        "--matlab-display-mode=nodesktop",
        "--matlab-session-mode=new",
        "--disable-telemetry=true",
        "--log-folder=/Users/antepil/Documents/ChatGPT/dc_dc_converter_neural_ode/.pi/mcp-logs"
      ],
      "cwd": "/Users/antepil/Documents/ChatGPT/dc_dc_converter_neural_ode",
      "lifecycle": "lazy-keep-alive",
      "requestTimeoutMs": 600000
    }
  }
}
```

- [ ] **Step 4: Validate both JSON files and resolved local paths**

Run:

```bash
python3 -m json.tool .mcp.json >/dev/null
python3 -m json.tool .pi/settings.json >/dev/null
test -x .tools/matlab-mcp-server/matlab-mcp-server
test -d .tools/matlab-agentic-toolkit/skills-catalog/matlab-core
```

Expected: All commands exit `0`.

- [ ] **Step 5: Commit only reviewable project configuration**

```bash
git add .mcp.json .pi/settings.json
git commit -m "chore: configure project-local MATLAB agent tools"
```

### Task 5: Reload and Verify MCP End to End

**Files:**
- Runtime log only, ignored: `.pi/mcp-logs/`

- [ ] **Step 1: Reload Pi configuration**

Run `/reload` in the active Pi session if `mcp({ connect: "matlab" })` does not yet recognize the newly added server.

Expected: MCP status lists both `atomicviz` and `matlab`.

- [ ] **Step 2: Connect the MATLAB MCP server**

Call:

```text
mcp({ connect: "matlab" })
```

Expected: The server connects and advertises `detect_matlab_toolboxes`, `check_matlab_code`, `evaluate_matlab_code`, `run_matlab_file`, and `run_matlab_test_file`.

- [ ] **Step 3: Detect MATLAB and installed toolboxes**

Call:

```text
mcp({ tool: "matlab_detect_matlab_toolboxes", args: {} })
```

Expected: The result reports MATLAB release `R2025a` and installed toolbox information.

- [ ] **Step 4: Evaluate a minimal MATLAB expression**

Call:

```text
mcp({
  tool: "matlab_evaluate_matlab_code",
  args: {
    "code": "fprintf('MCP_SMOKE_RESULT=%d\\n', 2 + 3);",
    "project_path": "/Users/antepil/Documents/ChatGPT/dc_dc_converter_neural_ode"
  }
})
```

Expected: Output includes `MCP_SMOKE_RESULT=5`.

- [ ] **Step 5: Confirm global configuration did not change**

Run:

```bash
for file in \
  "$HOME/.pi/agent/settings.json" \
  "$HOME/.codex/config.toml" \
  "$HOME/.claude/settings.json"; do
  if [ -f "$file" ]; then
    shasum -a 256 "$file"
  fi
done > /tmp/matlab-agentic-toolkit-install/global-config.after.sha256
diff -u \
  /tmp/matlab-agentic-toolkit-install/global-config.before.sha256 \
  /tmp/matlab-agentic-toolkit-install/global-config.after.sha256
```

Expected: `diff` exits `0` with no output.

- [ ] **Step 6: Run final repository checks**

Run:

```bash
python3 -m json.tool .mcp.json >/dev/null
python3 -m json.tool .pi/settings.json >/dev/null
git diff --check
git status --short
```

Expected: JSON validation and `git diff --check` pass; downloaded `.tools/` content and `.pi/mcp-logs/` do not appear in Git status.
