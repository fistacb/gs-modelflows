# Environment Setup Guide (`gs-modelflows`)

This guide helps humans and coding agents (Antigravity, Claude Code, Codex) set up all dependencies from scratch.
Run `python check_env.py` at any time to verify which flows are ready.

---

## 1. Python Environment (Required for all flows)

Python 3.10+ is required.

```bash
pip install -e ".[all]"
```

This installs `numpy`, `scipy`, `pillow`, `opencv-python`, `imageio`, `imageio-ffmpeg`, `scikit-image`, and `torch`.
For GPU acceleration in `gs-to-mesh`, install a CUDA-enabled PyTorch wheel matching your driver if `torch_cuda` is `false`.

---

## 2. Spark GPU Renderer (Required for `gs-modeling` QA & `gs-to-sprite`)

Requires **Node.js 18+** and **Google Chrome / Chromium**.

```bash
cd renderer/spark_demo
npm install
npm run sync-vendor
```

If you install skills globally (`python install_skills.py --target all`), set `GS_SPARK_DIR` to the absolute path of `renderer/spark_demo` so skills can locate `capture.mjs` from any workspace:

```bash
# Example (Linux/macOS)
export GS_SPARK_DIR="/path/to/gs-modelflows/renderer/spark_demo"

# Example (PowerShell)
$env:GS_SPARK_DIR = "C:\path\to\gs-modelflows\renderer\spark_demo"
```

---

## 3. Blender CLI (Required for `gs-to-mesh` & `gs-modeling` rig previews)

Install **Blender 4.2+** (tested on Blender 5.1) and ensure `blender` is on `PATH` or set `GS_BLENDER_BIN`.

---

## 4. Local ComfyUI + TripoSplat (Only required for `gs-modeling` Step 0: Image → `.ply`)

`gs-to-sprite` and `gs-to-mesh` work **without** ComfyUI (using any existing `.ply` file).
To generate new 3DGS `.ply` models from a 2D image via `step0_generate_ply.py`:

1. Install ComfyUI (`>= 0.41.0`) and start it on port `8188` (or set `GS_COMFYUI_URL` to match your port).
2. Install the **TripoSplat** ComfyUI nodes and download the weights referenced by `skills/gs-modeling/resources/triposplat_api_workflow.template.json`:
   - `dino_v3_vit_h.safetensors` (CLIP Vision)
   - `flux2-vae.safetensors` (VAE)
   - `triposplat_vae_decoder_fp16.safetensors` (TripoSplat VAE Decoder)
   - `triposplat_fp16.safetensors` (TripoSplat UNet)
3. (Optional) Install **BiRefNet** in ComfyUI if you prefer neural background removal over `step0_prep_image.py`.
4. (Optional) Set `GS_COMFYUI_DIR` to your ComfyUI root so `gs-to-mesh/scripts/extract_density_mesh.py` can use ComfyUI's GPU `_splat_density` kernel instead of the built-in PyTorch fallback.

---

## 5. Verify Setup

```bash
python check_env.py
```

Check `flow_readiness` in the JSON output to see which of `gs-modeling`, `gs-to-sprite`, and `gs-to-mesh` are ready to run.
