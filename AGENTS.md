# gs-modelflows Agent Instructions

This repository provides three modular **3D Gaussian Splatting (3DGS)** production skills and shared CLI tools that work across **Antigravity**, **Claude Code**, and **OpenAI Codex**.

## Available Skills (`skills/`)

1. **`gs-modeling`** (`skills/gs-modeling/SKILL.md`)
   - Single-image 3DGS generation (ComfyUI / TripoSplat), normalization, optional HD bust head transplant, dynamic doll eyes/eyelids & procedural mouth, 22-bone Harmonic DQS rigging, `v1/` package assembly, and Spark GPU QA video verification.
2. **`gs-to-sprite`** (`skills/gs-to-sprite/SKILL.md`)
   - Converts rigged or static 3DGS models into game-ready 2D / HD-2D sprite sheets (`.png`), transparent animated GIFs, and optional diorama previews using Spark GPU dual-background difference matting, telephoto/orthographic-like cameras, tinted 1px pixel outlines, and unified palettes.
3. **`gs-to-mesh`** (`skills/gs-to-mesh/SKILL.md`)
   - Converts 3DGS `.ply` models into watertight meshes and 3D-printable unified `.stl` / `.glb` files via 3D Gaussian density evaluation, morphological volume filling, marching cubes, Blender voxel remesh union, optional miniature circular base, and 6-view clay QA comparison.

## Repository Layout

- `shared/` — Agent-agnostic core Python modules (`ply_io.py`, `harmonic_dqs.py`, `procedural_mouth.py`, `config_loader.py`).
- `renderer/spark_demo/` — Headless Chrome + three.js + Spark GPU renderer (`capture.mjs`, `capture.js`, `capture.html`).
- `skills/` — Canonical skill definitions (Single Source of Truth).
- `.agents/skills/`, `.claude/skills/`, `.agent/skills/` — Auto-discovered copies kept in sync via `python install_skills.py --sync-repo`.

## Environment Variables (No Hardcoded Paths)

- `GS_COMFYUI_DIR` — Path to local ComfyUI installation (optional; accelerates `_splat_density`).
- `GS_COMFYUI_URL` — Local ComfyUI endpoint (default: `http://127.0.0.1:8188`).
- `GS_BLENDER_BIN` — Path to Blender CLI executable (default: `blender` on `PATH`).
- `GS_SPARK_DIR` — Path to `renderer/spark_demo` (recommended when skills are installed globally).
- `SPARK_CHROME` — Path to Chrome/Chromium binary for `renderer/spark_demo/capture.mjs` (auto-detected if omitted).
- `SPARK_SOFTWARE=1` — Force SwiftShader software WebGL fallback when no GPU is available.

Run `python check_env.py` first to inspect environment readiness. See `docs/SETUP.md` for full installation steps.

## Working Rules

- Never hardcode user-specific absolute paths or character-specific constants into scripts.
- Keep all generated `.ply`, `.stl`, `.blend`, `.png`, `.gif`, `.mp4`, and temporary frames inside a gitignored `work/<character>/` directory.
- When editing any file under `skills/`, run `python install_skills.py --sync-repo` to keep `.agents/skills/`, `.claude/skills/`, and `.agent/skills/` synchronized.
