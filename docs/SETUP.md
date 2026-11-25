English | [日本語](SETUP.ja.md)

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
2. Following the official ComfyUI TripoSplat tutorial ([https://docs.comfy.org/tutorials/3d/triposplat](https://docs.comfy.org/tutorials/3d/triposplat)), install the **TripoSplat** nodes and place the weights referenced by `skills/gs-modeling/resources/triposplat_api_workflow.template.json` under `ComfyUI/models/`:

| File | Save Path (under `ComfyUI/models/`) | Role | Download | License Reference | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `triposplat_fp16.safetensors` | `diffusion_models/` | TripoSplat UNet (diffusion model) | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat/blob/main/diffusion_models/triposplat_fp16.safetensors) | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat) | MIT |
| `triposplat_vae_decoder_fp16.safetensors` | `vae/` | TripoSplat VAE Decoder | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat/blob/main/vae/triposplat_vae_decoder_fp16.safetensors) | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat) | MIT |
| `dino_v3_vit_h.safetensors` | `clip_vision/` | CLIP Vision encoder | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat/blob/main/clip_vision/dino_v3_vit_h.safetensors) | Meta official ([facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) / [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)) | DINOv3 License (Meta custom license: commercial use allowed, military use prohibited, license must be included on redistribution) |
| `flux2-vae.safetensors` | `vae/` | VAE (latent conditioning) | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat/blob/main/vae/flux2-vae.safetensors) | Black Forest Labs official ([bfl.ai/blog/flux-2](https://bfl.ai/blog/flux-2) / [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2)) | Apache 2.0 (BFL official). Note: some mirrors list a non-commercial license (e.g. derived from FLUX.2-klein-9B) — **obtain from the official release or a source explicitly stating Apache 2.0** |
| `birefnet.safetensors` *(optional)* | `background_removal/` | Background removal | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat/blob/main/background_removal/birefnet.safetensors) | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | MIT |

*Please check the original license text for the latest terms.*

3. (Optional) Install **BiRefNet** in ComfyUI if you prefer neural background removal over `step0_prep_image.py`.
4. (Optional) Set `GS_COMFYUI_DIR` to your ComfyUI root so `gs-to-mesh/scripts/extract_density_mesh.py` can use ComfyUI's GPU `_splat_density` kernel instead of the built-in PyTorch fallback.

---

## 5. Verify Setup

```bash
python check_env.py
```

Check `flow_readiness` in the JSON output to see which of `gs-modeling`, `gs-to-sprite`, and `gs-to-mesh` are ready to run.
