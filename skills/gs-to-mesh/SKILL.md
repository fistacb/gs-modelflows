---
name: gs-to-mesh
description: >-
  Convert a 3D Gaussian Splatting (3DGS) .ply model into a watertight surface mesh and
  3D-printable unified .stl / .glb asset via 3D Gaussian density field evaluation,
  morphological solid volume filling, marching cubes, Blender voxel remesh union,
  optional miniature base attachment, manifold/Z=0 validation, and 6-view clay QA.
---

# 3DGS to Watertight Mesh & Unified STL Pipeline

## 目的
ポーズ適用済み（または静止）の 3D ガウシアンスプラット（`.ply`）から、**1 連結・非多様体エッジ 0・底面 Z=0 の水密（watertight）メッシュ**を抽出し、3D プリント向けの一体 `.stl` または 3D ゲーム向けの `.glb` として出力・検証する。

---

## 標準フロー（キャラクター非依存）

> **重要（3D プリント／単色クレイ向けの入力分岐）**:
> `gs-modeling` で制作したキャラクターを 3D プリントや単色メッシュ化する場合は、Stage 4 で可動目用に眼窩を彫り込んだ `05_rig/v1/master.ply` ではなく、**眼窩彫り込み前の `03_body/body.ply`** に `rig_skeleton.json` でポーズを適用した `.ply` を入力に使うこと。これにより、単色クレイ時に眼窩が骸骨状の穴や睨みスリットとして露出するのを防ぎ、未塗装フィギュア原型のような自然な顔プロポーションを保てる。

1. **ガウシアン密度場からのサーフェス抽出（`extract_density_mesh.py`）**
   - 低不透明度のハロー・浮遊ノイズを除外（既定 `opacity >= 0.20`）し、薄すぎるサーフェルを最小スケール（既定 `0.0016`）でわずかに肉付けしてサンプリングの隙間を塞ぐ。
   - 3D ボクセルグリッド（既定 `420` 分割）上で**異方性ガウシアン共分散（回転クォータニオン＋3軸スケール）**による正確な楕円体マハラノビス距離で密度場を評価する（`GS_COMFYUI_DIR` が指定されていれば ComfyUI の GPU `_splat_density` を使用し、未指定時は PyTorch ネイティブの異方性密度計算を使用）。※等方球近似はサーフェルがポップコーン状に膨らむ原因になるため使用しない。
   - **凹部保護ソリッド充填（`fill_volume`）**: 6 近傍の 3D モルフォロジー閉処理（既定 `closing=2`）→ 純粋な `binary_fill_holes` → 最大連結成分抽出で中実ボリュームを生成。※従来の 3 軸スキャンライン充填はフード奥や前髪・マント内側などの凹空間を誤って埋めてしまうため既定 OFF（必要時のみ `--scanline-fill`）。
   - 最小限の段差取りガウシアン平滑化（既定 `sigma=0.55`）後、Marching Cubes（`level=0.50`）でメッシュを抽出し `.npz`（＋任意で `.obj`）へ保存する。

2. **Blender ボクセル結合・エッジ保存バイラテラル法線ノイズ除去・一体 STL 出力（`assemble_watertight_stl.py`）**
   - 人物メッシュ（`--body-npz` / `--body-obj`）と、任意のプロップ（`--extra-mesh`）および円形台座（`--add-base`）を結合し、Blender の `REMESH` モディファイア（`VOXEL`, 既定 `voxel_size=0.0024`）で完全一体化する。
   - **浮遊ゴミ除去＆水密チェック**: 連結成分解析で極小の浮遊片を削除し、三角化＋法線再計算を行い、**非多様体エッジ（`nonmanifold_edges == 0`）** を検証する。
   - **エッジ保存バイラテラル法線フィルタリング（`bilateral_normal_denoise`）**: 既定 `--bilateral-iters 18 --bilateral-sigma 0.45 --bilateral-vert-iters 12` で、法線が近い滑らかな面（顔・フード・マント）のガタツキだけを滑らかにし、法線が大きく変化するディテール稜線（前髪・花飾り・リボン・スカートフリル・台座エッジ）は自動保存する。
   - **底面接地＆実寸スケール**: 底面をわずかに平坦クリップして `Z = 0` に接地させ、`--base-diameter-mm`（例：ミニチュア 1 インチ規格 `25.4 mm`）または `--target-height-mm` に合わせてスケーリングし、`.stl`（＋任意で `.glb`）と検証 JSON レポートを出力する。

3. **6 方向クレイ比較 QA（`--render-qa-dir`）**
   - Cycles クレイマテリアルで `0°, 45°, 90°, 180°, 225°, 270°` の 6 方向をレンダリングし、元の 3DGS スプラット（Spark レンダリング）と輪郭・ポーズ・接地・細部残存を比較確認する。

---

## 実行コマンド例

### Step 1: 3DGS `.ply` → 密度場サーフェスメッシュ（`.npz` / `.obj`）

```bash
python skills/gs-to-mesh/scripts/extract_density_mesh.py \
  --ply work/my-char/posed.ply \
  --out work/my-char/body_mesh.npz \
  --out-obj work/my-char/body_mesh.obj \
  --threshold 0.38 \
  --closing 3 \
  --grid-res 380
```

### Step 2: Blender 一体化・台座付与・水密 STL 出力＆6方向 QA

```bash
blender -b --python skills/gs-to-mesh/scripts/assemble_watertight_stl.py -- \
  --body-npz work/my-char/body_mesh.npz \
  --out-stl work/my-char/model_25p4mm.stl \
  --out-glb work/my-char/model_25p4mm.glb \
  --add-base \
  --base-diameter-mm 25.4 \
  --voxel-size 0.0032 \
  --render-qa-dir work/my-char/qa_mesh
```

---

## キャラクター固有設定と再利用処理の分離ルール

- **再利用する処理（本スキル）**: 密度場評価、ボリューム充填、Marching Cubes、ボクセル結合、浮遊片除去、水密・底面 `Z=0`・外寸検証、6 方向クレイ撮影。
- **キャラクター固有の設定（作業フォルダ側で指定）**: 入力 `.ply` パス、ポーズ・武器などの追加メッシュ（`--extra-mesh`）、台座の有無・径（`--base-diameter-mm`）や目標身長（`--target-height-mm`）、座標系変換（`--coord-preset blender|raw_to_blender`）。

---

## 検証チェックリスト（出力レポート JSON）
- `nonmanifold_edges == 0`（完全な閉じた水密メッシュ）
- `components_after_cleanup == 1`（1 連結の一体モデル）
- `min_z_mm == 0.0`（プリントベッドへ確実に接地）
- `overall_dimensions_mm` が目標サイズと一致していること
- 6 方向クレイ比較で主要な輪郭・ポーズ・腕や武器の接続が崩れていないこと
