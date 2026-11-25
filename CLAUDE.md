# CLAUDE.md

This repository follows the shared instructions in `AGENTS.md`.
Read `AGENTS.md` first, then load the relevant skill under `.claude/skills/` (canonical source in `skills/`):

- `.claude/skills/gs-modeling/SKILL.md` — 3DGS character modeling, dynamic eyes/mouth, and Harmonic DQS rigging
- `.claude/skills/gs-to-sprite/SKILL.md` — 3DGS to 2D / HD-2D sprite sheets & transparent GIFs
- `.claude/skills/gs-to-mesh/SKILL.md` — 3DGS to Gaussian density surface mesh & watertight STL/GLB

Key reminders:
- Use environment variables (`GS_COMFYUI_DIR`, `GS_COMFYUI_URL`, `GS_BLENDER_BIN`, `SPARK_CHROME`) instead of hardcoded paths.
- After editing files in `skills/`, run `python install_skills.py --sync-repo`.
