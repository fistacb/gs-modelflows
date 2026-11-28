# Modular Skeleton Schema v2.1 Guide (`gs-modelflows`)

`schema_version: "2.1"` extends the existing `2.0` humanoid `rig_skeleton.json` with **100% backward compatibility** so that existing `v1/` packages (e.g. `hood-01`), Spark GPU `SplatSkinning` (up to 256 bones), and downstream skills (`gs-to-sprite`, `gs-to-mesh`, `gs-stage`) continue to work without modification.

## 1. Key Modules

| Module | Role |
|---|---|
| `shared/harmonic_dqs.py` | Dual-mode Harmonic Graph-Laplacian skinning: legacy v2.0 humanoid path + generic v2.1 group-isolated & `seam_cut` path |
| `shared/secondary_motion.py` | `SecondaryMotionSolver`: bone-length-normalized spring-damper physics with material presets (`ear`/`tail`/`cape`/`hair`/`ribbon`) and lateral cloth-sheet coupling |
| `shared/procedural_locomotion.py` | Analytical 2-Bone IK, duty-factor quadruped gaits (`walk`/`trot`/`gallop`) with head stabilization, and asymmetric DSR avian wingbeat flight |
| `shared/skeleton_builder.py` | Hybrid (auto-detect + recipe/override) compiler for `humanoid`, `quadruped`, `custom` archetypes and composable attachments |
| `skills/gs-modeling/scripts/build_modular_rig.py` | CLI to build/extend `rig_skeleton.json` from a PLY + recipe JSON and verify Harmonic weights |
| `skills/gs-modeling/scripts/render_modular_qa_video.py` | Archetype-agnostic Spark GPU QA video & 3x3 contact sheet generator (`--mode standard \| wing_showcase \| animal_locomotion`) |

## 2. Schema v2.1 Optional Fields

Top-level fields (all optional; omitting them behaves identically to v2.0):
- `"archetype"`: `"humanoid" | "quadruped" | "custom"`
- `"seam_cuts"`: list of kNN graph cut planes (`{origin, normal, radius?, bbox_min?, bbox_max?}`)
- `"attachments"`: record of recipe attachments used to build the rig

Per-bone optional fields inside `bones[name]`:
- `"skin_group"`: isolates Harmonic core pinning between distinct anatomical groups (e.g. `"wing_R"`, `"tail"`, `"jaw"`, `"cape_back"`)
- `"role"`: `"root" | "pelvis" | "spine" | "chest" | "neck" | "head" | "jaw" | "limb" | "wing" | "tail" | "ear" | "cloth" | "ribbon" | "hair" | "accessory"`
- `"side_lock"`: `"L" | "R"` to prevent cross-midline weight bleed
- `"rigid"`: `true` to lock rigid cranium / hard accessories to weight `1.0`
- `"pin_radius"` / `"influence_radius_scale"` / `"region_box"`: fine-grained Harmonic influence control
- `"seam_cut"`: per-bone mouth/seam plane cut (`keep_side: -1 | +1`) so closed 3DGS muzzles open cleanly without stretching lips
- `"dynamics"`: secondary-motion spring-damper parameters:
  ```json
  {
    "stiffness": 18.0,
    "damping": 0.25,
    "gravity": 0.20,
    "wind_scale": 1.0,
    "inertia": 1.0,
    "max_angle_deg": 32.0
  }
  ```

## 3. Hybrid Workflow (Auto-Detect + Recipe + Overrides)

Example recipe (`rig_recipe.json`) for a winged quadruped with muzzle jaw, ears, tail, and chest ribbons:

```json
{
  "archetype": "quadruped",
  "attachments": [
    {"type": "jaw", "max_open_deg": 35.0},
    {"type": "wing", "parent": "chest", "side": "both", "n_segments": 3},
    {"type": "tail", "parent": "pelvis", "n_segments": 3},
    {"type": "ear", "parent": "head", "n_segments": 2},
    {"type": "ribbon", "parent": "chest", "n_segments": 2}
  ],
  "custom_bones": {},
  "overrides": {}
}
```

CLI usage:
```powershell
python skills/gs-modeling/scripts/build_modular_rig.py --ply normalized.ply --recipe rig_recipe.json --out rig_skeleton.json --verify-weights
python skills/gs-modeling/scripts/render_modular_qa_video.py --ply normalized.ply --skeleton rig_skeleton.json --spark-dir renderer/spark_demo --out-mp4 qa.mp4 --out-sheet qa_sheet.png
```
