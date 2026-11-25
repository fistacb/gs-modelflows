[English](SETUP.md) | 日本語

# 環境構築ガイド（`gs-modelflows`）

人間およびコーディングエージェント（Antigravity / Claude Code / Codex）が、ゼロから依存環境をセットアップするためのガイドです。
どのフローが実行可能な状態かは、いつでも `python check_env.py` で確認できます。

---

## 1. Python 環境（全フロー必須）

Python 3.10 以上が必要です。

```bash
pip install -e ".[all]"
```

これにより `numpy`, `scipy`, `pillow`, `opencv-python`, `imageio`, `imageio-ffmpeg`, `scikit-image`, `torch` がインストールされます。
`gs-to-mesh` で GPU 高速化を使う場合、`check_env.py` で `torch_cuda` が `false` であれば、お使いのドライバに合った CUDA 対応版の PyTorch をインストールしてください。

---

## 2. Spark GPU レンダラー（`gs-modeling` QA・`gs-to-sprite` で必須）

**Node.js 18 以上** および **Google Chrome / Chromium** が必要です。

```bash
cd renderer/spark_demo
npm install
npm run sync-vendor
```

スキルをグローバルにインストールした場合（`python install_skills.py --target all`）は、別ワークスペースからでも `capture.mjs` を参照できるよう、環境変数 `GS_SPARK_DIR` に `renderer/spark_demo` の絶対パスを設定してください：

```bash
# Example (Linux/macOS)
export GS_SPARK_DIR="/path/to/gs-modelflows/renderer/spark_demo"

# Example (PowerShell)
$env:GS_SPARK_DIR = "C:\path\to\gs-modelflows\renderer\spark_demo"
```

---

## 3. Blender CLI（`gs-to-mesh`・`gs-modeling` リグプレビューで必須）

**Blender 4.2 以上**（Blender 5.1 で動作確認済み）をインストールし、`blender` に `PATH` を通すか、環境変数 `GS_BLENDER_BIN` で実行ファイルのパスを指定してください。

---

## 4. ローカル ComfyUI + TripoSplat（`gs-modeling` Step 0：画像 → `.ply` のみで必須）

`gs-to-sprite` と `gs-to-mesh` は既存の `.ply` があれば ComfyUI **なし** で動作します。
2D 画像から `step0_generate_ply.py` で新しく 3DGS `.ply` を生成する場合は、以下の手順で準備します：

1. ComfyUI（`>= 0.41.0`）をインストールし、ポート `8188` で起動します（別ポートを使う場合は `GS_COMFYUI_URL` を合わせてください）。
2. ComfyUI 公式の TripoSplat チュートリアル（[https://docs.comfy.org/tutorials/3d/triposplat](https://docs.comfy.org/tutorials/3d/triposplat)）を参考に、**TripoSplat** 用ノードおよび `skills/gs-modeling/resources/triposplat_api_workflow.template.json` が参照するモデル重みを配置します：

| ファイル名 | 役割 | 入手元 | ライセンス |
| :--- | :--- | :--- | :--- |
| `triposplat_fp16.safetensors` | TripoSplat UNet（拡散モデル本体） | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat/blob/main/diffusion_models/triposplat_fp16.safetensors) | MIT |
| `triposplat_vae_decoder_fp16.safetensors` | TripoSplat VAE デコーダー | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat/blob/main/vae/triposplat_vae_decoder_fp16.safetensors) | MIT |
| `dino_v3_vit_h.safetensors` | CLIP Vision エンコーダー | [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)（ComfyUI チュートリアル案内ミラー: [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat/blob/main/clip_vision/dino_v3_vit_h.safetensors)） | DINOv3 License（Meta 独自ライセンス。商用利用は可、軍事用途は禁止、再配布時はライセンスの同梱が必要） |
| `flux2-vae.safetensors` | VAE（潜在表現エンコード） | Black Forest Labs 公式（[black-forest-labs](https://huggingface.co/black-forest-labs)）または ComfyUI チュートリアル案内先（[docs.comfy.org/tutorials/3d/triposplat](https://docs.comfy.org/tutorials/3d/triposplat)） | Apache 2.0（BFL 公式）。※ミラーによっては非商用ライセンス（FLUX.2-klein-9B 由来など）と表記されているため、**公式または Apache 2.0 と明記された配布元から入手すること** |
| `birefnet.safetensors`（任意） | 背景除去（ニューラル前処理） | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)（ComfyUI チュートリアル案内ミラー: [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat/blob/main/background_removal/birefnet.safetensors)） | MIT |

※各ライセンスの最新の条件は原文を確認してください。

3. （任意）`step0_prep_image.py` の代わりにニューラル背景除去を使いたい場合は、ComfyUI に **BiRefNet** を導入してください。
4. （任意）環境変数 `GS_COMFYUI_DIR` に ComfyUI のルートディレクトリを設定すると、`gs-to-mesh/scripts/extract_density_mesh.py` が組み込みの PyTorch フォールバックの代わりに ComfyUI の GPU `_splat_density` カーネルを利用します。

---

## 5. セットアップの確認

```bash
python check_env.py
```

出力 JSON 内の `flow_readiness` を確認し、`gs-modeling`・`gs-to-sprite`・`gs-to-mesh` のどれが実行可能になっているか確認してください。
