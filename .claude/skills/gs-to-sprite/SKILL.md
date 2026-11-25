---
name: gs-to-sprite
description: >-
  Convert rigged or static 3D Gaussian Splatting (3DGS) assets into game-ready 2D / HD-2D
  transparent sprite sheets (.png), animated .gif files, and optional diorama previews using
  dual-background difference matting, telephoto/top-down cameras, tinted 1px outlines, and
  unified flicker-free palettes.
---

# 3DGS to 2D / HD-2D Game Sprite Pipeline

## 目的
リギング済み 3DGS（`anim.json`）または静止 3DGS（`.ply`）から、**多視点で絵柄が一貫したゲーム用透過スプライトシート（PNG）および透過アニメーション GIF** を自動生成する。

---

## 解像度ガイドライン（頭身に応じた選定）

| キャラクター頭身 | 推奨キャンバスサイズ | 想定ジャンル・見え方 |
| :--- | :--- | :--- |
| **2〜3 頭身（ちび・デフォルメ）** | `64×64` 〜 `80×80` px | クラシック HD-2D RPG（オクトパストラベラー系） |
| **4〜5 頭身（中頭身）** | `112×112` 〜 `144×144` px | 高精細クォータービュー RPG / SRPG |
| **6〜8 頭身（高頭身・リアル寄り）** | `160×160` 〜 `224×224` px | ハイレゾ HD-2D / 2D アクション RPG / 格闘ゲーム風（目元や小物が潰れない） |

---

## 核心テクニック（品質を担保する 4 原則）

1. **黒・白 2 パス差分マッティング（フリンジゼロの完全透過）**
   - 3DGS は半透明スプラットの集合体であるため、単色クロマキーでは輪郭に白・黒のフリンジが残る。
   - 同一カメラ・同一ポーズで `#000000` と `#ffffff` の 2 枚を Spark GPU で描画し、`alpha = 1 - mean(I_white - I_black)`, `rgb = I_black / alpha` で数学的に厳密な RGBA を復元する。
2. **望遠＋やや俯瞰カメラ（HD-2D パース）**
   - `fov = 10°〜12°`（望遠で平行投影に近い歪みのない見え方）、`pitch = 10°〜15°`（見下ろし感）を標準とする。
3. **共通バウンディングボックス＋足元接地＋色付き 1px アウトライン**
   - 全方向・全フレームの共通バウンディングボックスでクロップし、足元を下端から一定マージンに揃える。
   - ドット絵らしいパキッとした二値アルファの上から、隣接ピクセル色を暗くした **暖色系ダーク 1px アウトライン** を自動付与する（真っ黒より 3D 背景に自然に馴染む）。
4. **明部優先の統一パレット減色（チラつき防止）**
   - 全フレーム・全方向で 1 つの共通パレットを生成することで、フレーム間の色チラつきを防ぐ。
   - 暗い衣装の面積が大きいモデルでも顔・肌・白シャツ・金属ハイライトがくすまないよう、高輝度ピクセルに重みを付けてパレットを割り当てる。

---

## 実行コマンド例

### リギング済み 3DGS（`anim.json`）から 4 方向歩きスプライトを生成

```bash
python skills/gs-to-sprite/scripts/render_hd2d_sprites.py \
  --anim-json path/to/anim.json \
  --spark-dir renderer/spark_demo \
  --out-dir work/my-char/sprites \
  --frame-prefix loco- \
  --num-frames 12 \
  --size 160 \
  --directions front:0,diag:40,side:90,back:180 \
  --scene-preview
```

### 静止 3DGS（`.ply`）から 8 方向ターンテーブル・スプライトを生成

```bash
python skills/gs-to-sprite/scripts/render_hd2d_sprites.py \
  --ply work/my-prop/model.ply \
  --spark-dir renderer/spark_demo \
  --out-dir work/my-prop/sprites \
  --size 128 \
  --directions d000:0,d045:45,d090:90,d135:135,d180:180,d225:225,d270:270,d315:315
```

### 歩行モーション確認のチェックポイント
- **ムーンウォーク防止**: 接地脚が後ろへ蹴り出し、遊脚が膝を曲げて前へ振り出されているか確認する（位相が逆の場合は `--reverse-frames` を指定）。
- **背景スクロール整合**: キャラが画面左向きなら背景は右へ、右向きなら左へスクロールさせる。
- **目元・シルエット確認**: 生成される `comparison_sheet.png` の顔ズーム枠で、目・アクセサリー・足元が潰れていないか確認し、必要なら `--size` を上げる。
