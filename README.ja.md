[English](README.md) | 日本語

# gs-modelflows

ローカル環境で完結する **3D Gaussian Splatting（3DGS）** アセット制作パイプラインと、**マルチエージェント対応スキル**（**Antigravity** / **Claude Code** / **OpenAI Codex** 互換）をまとめたリポジトリです。

1 枚の 2D リファレンス画像からクリーンなリグ付き 3DGS キャラクターを作成し、用途に応じて以下の 3 つのフローへ展開できます。

1. **3DGS モデリング＆リギング（`gs-modeling`）** — 単一画像からの 3DGS 生成（ComfyUI / TripoSplat）、正規化、シード比較、素体の整理、動的ドールアイ／まぶた（必須）、プロシージャルなリップシンク口（必須）、22 ボーンの Harmonic DQS リギング、Spark GPU による QA 動画生成。
2. **2D / HD-2D ゲームスプライト化（`gs-to-sprite`）** — 黒・白 2 パス差分マッティングによる完全透過、望遠・やや俯瞰の HD-2D カメラ、色付き 1px ピクセルアウトライン、チラつきのない統一パレット減色による多方向透過スプライトシート（`.png`）およびアニメーション `.gif` 出力。
3. **水密メッシュ＆一体 STL 出力（`gs-to-mesh`）** — 3D ガウシアン密度場の抽出、モルフォロジーによるソリッド充填、Marching Cubes、Blender ボクセル結合（任意のミニチュア台座付与）、非多様体・底面 Z=0 検証、6 方向クレイ比較。

---

## Architecture

```text
gs-modelflows/
├── AGENTS.md                   # Codex / 汎用エージェント向け指示書
├── CLAUDE.md                   # Claude Code 向け指示書
├── install_skills.py           # リポジトリ内スキル同期 & Antigravity / Claude / Codex へのグローバル配置
├── shared/                     # 共通 Python コア（PLY I/O, Harmonic DQS, プロシージャル口, 設定ローダー）
├── renderer/spark_demo/        # ヘッドレス three.js + Spark GPU SplatSkinning レンダラー
├── skills/                     # スキル本体（Single Source of Truth）
│   ├── gs-modeling/
│   ├── gs-to-sprite/
│   └── gs-to-mesh/
├── .agents/skills/             # Codex / Agent Skills 対応ホストによる自動検出用
├── .claude/skills/             # Claude Code による自動検出用
└── .agent/skills/              # Antigravity による自動検出用
```

---

## Quick Start

### 1. Python・レンダラーのセットアップ

手順の詳細は [docs/SETUP.ja.md](docs/SETUP.ja.md)（英語版: [docs/SETUP.md](docs/SETUP.md)）を参照してください（任意のローカル ComfyUI + TripoSplat の導入手順も含みます）。

```bash
pip install -e ".[all]"
cd renderer/spark_demo && npm install && npm run sync-vendor
cd ../..
python check_env.py
```

任意の環境変数（利用可能な場合は既定値で自動検出されます）：

| 環境変数 | 用途 | 既定値 |
| :--- | :--- | :--- |
| `GS_COMFYUI_DIR` | ローカル ComfyUI リポジトリのルート（`_splat_density` 高速化用） | *（未設定時は PyTorch フォールバックを使用）* |
| `GS_COMFYUI_URL` | ローカル ComfyUI の API エンドポイント | `http://127.0.0.1:8188` |
| `GS_BLENDER_BIN` | Blender CLI 実行ファイルのパス（`>= 4.2` / `5.x`） | `blender` |
| `GS_SPARK_DIR` | `renderer/spark_demo` のパス（グローバルスキル利用時に推奨） | 自動検出 |
| `SPARK_CHROME` | ヘッドレス WebGL 用 Chrome / Chromium 実行ファイルのパス | 自動検出 |
| `SPARK_SOFTWARE` | `1` を指定すると SwiftShader によるソフトウェア WebGL を強制 | `0` |

> **プラットフォームと言語に関する補足**:
> - 主に **Windows 11（NVIDIA GPU）** で動作確認しています。macOS / Linux でもクロスプラットフォーム向け WebGL フラグで動作します（`SPARK_SOFTWARE=1` のフォールバックも利用可能）。
> - トップレベルのドキュメントは英語（`README.md`, `AGENTS.md`, `docs/SETUP.md`）と日本語（`README.ja.md`, `docs/SETUP.ja.md`）の両方を用意しており、各 `SKILL.md` の詳細手順は日本語で記述されています（対応する各コーディングエージェントからそのまま読み込めます）。

### 2. Antigravity / Claude Code / Codex での利用

- **プロジェクト単位で利用（設定不要）**: `git clone` して Antigravity / Claude Code / Codex で `gs-modelflows` を開くだけで、`.agent/skills/`・`.claude/skills/`・`.agents/skills/` が自動的に認識されます。
- **グローバルにインストール（別ワークスペースからも利用）**:

```bash
# Install all 3 skills for all supported agents
python install_skills.py --target all

# Or target a specific agent
python install_skills.py --target claude
python install_skills.py --target codex
python install_skills.py --target antigravity
```

---

## 3 つのフロー概要

### Flow 1: `gs-modeling`（画像 → リグ付き 3DGS）
- クリーンな透過入力画像の準備（`step0_prep_image.py`, `gate0_check_input.py`）
- 3DGS `.ply` の生成と正規化（`step0_generate_ply.py`, `step1_normalize.py`）
- スケルトン合わせ、眼窩くり抜き、動的眼球・まぶた＆プロシージャル口の構築（`step2_fit_skeleton.py` .. `step5_merge_ply.py`）
- `v1/` パッケージ凍結＆ GPU スキニング QA 動画レンダリング（`step7_build_v1_package.py`, `render_standard_qa_video.py`, `verify_gate.py`）

### Flow 2: `gs-to-sprite`（3DGS → 2D / HD-2D スプライト＆ GIF）
```bash
python skills/gs-to-sprite/scripts/render_hd2d_sprites.py \
  --anim-json path/to/anim.json \
  --out-dir work/my-char/sprites \
  --size 160 \
  --directions front:0,diag:40,side:90,back:180 \
  --scene-preview
```
- 任意のスプライト解像度に対応（`64`, `80`, `128`, `160`, `224` など）
- 黒・白 2 パス（`#000000` / `#ffffff`）の差分マッティングで 3DGS 特有のエッジのフリンジを完全に除去
- 明部を優先した統一パレット減色により、フレーム間の色チラつきを防止

### Flow 3: `gs-to-mesh`（3DGS → 密度場メッシュ → 水密 STL / GLB）
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

MIT License — 詳細は [LICENSE](LICENSE) を参照してください。

> **サードパーティ製モデル・ツールについて**: 任意のワークフローから参照する外部モデルおよびツール（TripoSplat、BiRefNet、ComfyUI、Blender、three.js、`@sparkjsdev/spark` など）は本リポジトリには同梱されておらず、それぞれのライセンスに従います。
