# Project-Local MATLAB Agentic Toolkit Design

## Goal

Install the official MATLAB Agentic Toolkit and MATLAB MCP Server for this repository only. No user-global or system-global agent, MCP, Codex, Pi, or MATLAB configuration may be changed.

## Scope

- Use MATLAB R2025a at `/Volumes/MacApp/MATLAB_R2025a.app`, matching the release that created the repository's Simulink model.
- Install the current official MATLAB Agentic Toolkit release under `.tools/matlab-agentic-toolkit/`.
- Install the matching official Apple Silicon MATLAB MCP Server binary under `.tools/matlab-mcp-server/`.
- Enable only the required `matlab-core` skill group through project-local Pi settings.
- Register the `matlab` stdio server in the repository's existing `.mcp.json` without changing the existing `atomicviz` entry.
- Disable MATLAB MCP telemetry and use this repository as MATLAB's initial working folder.

## Project Layout

- `.tools/matlab-agentic-toolkit/` — project-local Toolkit checkout.
- `.tools/matlab-mcp-server/matlab-mcp-server` — project-local executable.
- `.pi/settings.json` — project-local Pi skill discovery configuration.
- `.mcp.json` — project-local shared MCP server configuration.
- `.gitignore` — excludes downloaded Toolkit files, MCP binary, and local MCP logs.

The downloaded executables and Toolkit checkout remain local and are not committed. The small configuration files remain reviewable in the repository.

## Runtime Configuration

The `matlab` MCP server will:

- use stdio transport;
- run the project-local server binary;
- pass `--matlab-root=/Volumes/MacApp/MATLAB_R2025a.app`;
- pass the repository root as `--initial-working-folder`;
- use lazy startup;
- disable telemetry;
- allow a long request timeout suitable for MATLAB startup and Simulink work.

The project-local Pi settings will add only `.tools/matlab-agentic-toolkit/skills-catalog/matlab-core` to skill discovery. No symlink or file will be created under `~/.agents`, `~/.pi`, `~/.codex`, or another project.

## Verification

1. Confirm the downloaded MCP binary reports its version.
2. Confirm the Toolkit checkout is pinned to an official release or commit.
3. Validate `.mcp.json` and `.pi/settings.json` as JSON.
4. Connect Pi's MCP adapter to the `matlab` server after configuration reload.
5. Call `detect_matlab_toolboxes` and confirm MATLAB R2025a.
6. Call `evaluate_matlab_code` with `2 + 3` and require the result `5`.
7. Check Git status to confirm no user-global configuration was modified by the installation.

## Failure Handling

- If GitHub download fails, stop without partially registering a missing binary.
- If macOS quarantine blocks the binary, remove quarantine only from the project-local binary.
- If MATLAB startup or licensing fails, preserve logs locally and report the exact failure without changing another MATLAB installation.
- If the current Pi session cannot hot-reload project configuration, request one `/reload` and continue verification afterward.
