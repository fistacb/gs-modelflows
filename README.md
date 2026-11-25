# gs-modelflows

Modular local **3D Gaussian Splatting (3DGS)** asset pipelines and **multi-agent skills** (compatible with **Antigravity**, **Claude Code**, and **OpenAI Codex**).

Start from a single 2D reference image, build a clean rigged 3DGS character, and branch into:
1. **3DGS Modeling & Rigging (`gs-modeling`)** — Single-image 3DGS generation (ComfyUI / TripoSplat), normalization, optional HD bust head transplant, dynamic doll eyes/eyelids, procedural lip-sync mouth, 22-bone Harmonic DQS rigging, and Spark GPU QA videos.
2. **2D / HD-2D Game Sprites (`gs-to-sprite`)** — Multi-angle transparent sprite sheets (`.png`) and animated `.gif` exports with dual-background exact alpha matting, telephoto/top-down framing, tinted 1px pixel outlines, and unified flicker-free palettes.
3. **Watertight Mesh & Unified STL (`gs-to-mesh`)** — 3D Gaussian density field extraction, morphological solid filling, marching cubes, Blender voxel union (with optional miniature base), manifold/Z=0 validation, and 6-view clay comparison.

---

## Architecture

```text
gs-modelflows/
├── AGENTS.md                   # Codex / generic agent instructions
├── CLAUDE.md                   # Claude Code instructions
├── install_skills.py           # Sync repo skill dirs & install globally for Antigravity / Claude / Codex
├── shared/                     # Core Python modules (PLY I/O, Harmonic DQS, procedural mouth, config)
├── renderer/spark_demo/        # Headless three.js + Spark GPU SplatSkinning renderer
├── skills/                     # Canonical skill sources (Single Source of Truth)
│   ├── gs-modeling/
│   ├── gs-to-sprite/
│   └── gs-to-mesh/
├── .agents/skills/             # Auto-discovered by Codex / Agent Skills hosts
├── .claude/skills/             # Auto-discovered by Claude Code
└── .agent/skills/              # Auto-discovered by Antigravity
```

---

## Quick Start

### 1. Python & Renderer Setup

```bash
pip install -e ".[all]"
```

Set up the Spark GPU renderer vendor dependencies (three.js + `@sparkjsdev/spark`):

```bash
cd renderer/spark_demo
npm install
npm run sync-vendor
```

Optional environment variables (defaults auto-detect when available):

| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `GS_COMFYUI_DIR` | Local ComfyUI repository root (for TripoSplat / `_splat_density`) | *(unset)* |
| `GS_COMFYUI_URL` | Local ComfyUI API endpoint | `http://127.0.0.1:8188` |
| `GS_BLENDER_BIN` | Path to Blender CLI executable (`>= 4.2` / `5.x`) | `blender` |
| `SPARK_CHROME` | Path to Chrome/Chromium executable for headless WebGL | auto-detected |
| `SPARK_SOFTWARE` | Set to `1` to force SwiftShader software WebGL | `0` |

### 2. Using with Antigravity, Claude Code, or Codex

- **Project-local (zero setup)**: Just `git clone` and open `gs-modelflows` in Antigravity, Claude Code, or Codex — the `.agent/skills/`, `.claude/skills/`, and `.agents/skills/` directories are discovered automatically.
- **Global install (use from any workspace)**:

```bash
# Install all 3 skills for all supported agents
python install_skills.py --target all

# Or target a specific agent
python install_skills.py --target claude
python install_skills.py --target codex
python install_skills.py --target antigravity
```

---

## The Three Flows at a Glance

### Flow 1: `gs-modeling` (Image → Rigged 3DGS)
- Prepare clean alpha input (`step0_prep_image.py`, `gate0_check_input.py`)
- Generate & normalize 3DGS `.ply` (`step0_generate_ply.py`, `step1_normalize.py`)
- Fit skeleton, carve eye sockets, build dynamic eyeballs/eyelids & procedural mouth (`step2_fit_skeleton.py` .. `step5_merge_ply.py`)
- Freeze `v1/` package & render GPU-skinned QA video (`step7_build_v1_package.py`, `render_standard_qa_video.py`, `verify_gate.py`)

### Flow 2: `gs-to-sprite` (3DGS → 2D / HD-2D Sprites & GIFs)
```bash
python skills/gs-to-sprite/scripts/render_hd2d_sprites.py \
  --anim-json path/to/anim.json \
  --out-dir work/my-char/sprites \
  --size 160 \
  --directions front:0,diag:40,side:90,back:180 \
  --scene-preview
```
- Supports arbitrary sprite resolutions (`64`, `80`, `128`, `160`, `224`, etc.)
- Dual-background (`#000000` / `#ffffff`) difference matting removes 3DGS edge fringe completely
- Unified palette quantization with highlight preservation prevents frame-to-frame flicker

### Flow 3: `gs-to-mesh` (3DGS → Density Mesh → Watertight STL / GLB)
```bash
# 1. Extract watertight surface mesh from 3DGS Gaussian density field
python skills/gs-to-mesh/scripts/extract_density_mesh.py \
  --ply work/my-char/posed.ply \
  --out work/my-char/body_mesh.npz \
  --threshold 0.33 --closing 2 --grid-res 380

# 2. Voxel union + optional miniature base + manifold check + STL/GLB export + 6-view QA
blender -b --python skills/gs-to-mesh/scripts/assemble_watertight_stl.py -- \
  --body-npz work/my-char/body_mesh.npz \
  --out-stl work/my-char/model_25p4mm.stl \
  --add-base --base-diameter-mm 25.4 \
  --render-qa-dir work/my-char/qa_mesh
```

---

## License

MIT License — see [LICENSE](LICENSE).
