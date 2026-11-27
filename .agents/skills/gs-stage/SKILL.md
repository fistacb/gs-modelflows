---
name: gs-stage
description: >-
  3D Gaussian Splatting (3DGS) map & stage artwork pipeline built on decoupled Frame (multi-layer 3D
  volumetric structure, walkable spans, prop anchors, lights), Theme (swappable surface masonry/terrain,
  blending, foliage, atmosphere, lighting), and standalone Splat objects (`v1_prop/` + rigged `v1/`).
  Compiles true multi-layer 3D navigation surfaces (`(ix, iy, span_idx)` with `max_step_height` and
  vertical headroom for underpass/overpass 3D crossings), plans slope-aware character walk trajectories
  for both 3DGS walkthrough videos and 2D/HD-2D games, and exports decoupled outputs (3DGS `.ply`,
  multi-angle stills, Depth/Normal maps, and walkthrough `.gif`). Use when assembling 3DGS maps,
  dioramas, multi-story stages, or character walkthroughs on 3D terrain.
---

# `gs-stage` — 立体フレーム × テーマ × スプラットによるマップアートワーク制作

## 基本概念（データと描画の分離）

マップアートワークを以下の 3 要素に分離して制作し、最終的な出力形式（3DGS・静止画・深度／法線マップ・動画・ゲーム用ナビデータ）を制作データから切り離す。

1. **フレーム（`frame.json`）**: 空間の構造・意味・配置のみを記述する（外観品質を持たない）。
   - `ground`: ベース地形の高低・領域（`forest_floor`, `stone_path`, `water` 等）
   - `volumes`: 立体構造物（`platform` テラス、`overpass` 高架橋、`ramp` 階段スロープ、`pillar` 柱）。**2.5D の単一ハイトマップではなく多層 3D スパン（`spans[iy][ix]`）としてコンパイル**するため、同じ `(x, y)` に 1 階（高架下）と 2 階（石橋）が上書きされず共存する。
   - `props`: `gs-modeling`（`mode: static_prop`）で作った独立 `v1_prop/` の配置先（`layer` 指定で 1 階・2 階それぞれの床へ自動接地）。
   - `lights`: 環境・点光源アンカー。
2. **テーマ（`theme.json`）**: スプラット外観・石積み目地・苔の馴染ませ・大気（霧・光の粒）・ライティングのルール集。**同一の `frame.json` に別の `theme.json` を差し替えるだけで構造を保ったまま別景観へ切り替え可能**。
3. **3D 歩行可能フロア ＆ 経路計画（`volumetric_frame_3d.py`）**:
   - 各セルの多層スパンに対し、`abs(z_next - z_cur) <= max_step_height`（既定 `0.36m`）かつ頭上クリアランス `>= 1.55m` の隣接のみ接続する **真の 3D A\*（`(ix, iy, span_idx)`）** を実行。
   - 動画生成時の 22 ボーン Harmonic DQS 歩行（勾配前傾・階段膝上げ・立体交差）と、2D/HD-2D ゲームのスプライト接地・前後ソート・足音種別判定の両方に共通の指標として使える。

---

## 標準ワークフロー

### 1. フレーム ＆ テーマの準備
テンプレートをコピーして編集する：
- フレーム雛形: [frame_3d.template.json](./resources/frame_3d.template.json)
- 森＆石造遺跡テーマ雛形: [theme_forest_ruins.json](./resources/theme_forest_ruins.json)

### 2. ステージ統合アセンブル ＆ 多層 3D ナビ検証（`assemble_stage.py`）
```bash
python skills/gs-stage/scripts/assemble_stage.py \
  --frame work/stage_ruins/frame_ruins_3d.json \
  --theme work/stage_ruins/theme_mossy_ruins.json \
  --props-root work/forest_pack \
  --out-dir work/stage_ruins \
  --waypoints "0,-3.4,0; 0,0,0; -1.6,2.6,0; 2.1,2.6,2.05; 1.8,0,2.05; 0,0,2.19; -2.5,0,2.05" \
  --spark-dir <spark_demo> \
  --export-maps
```

#### 主な出力
- `map_<theme>.ply`: 統合 3DGS マップ（Blender Z-up・実寸 m）
- `nav_surface.json` / `nav_multilayer_3d_preview.png`: 多層 3D 歩行可能フロアデータ＆Layer 0 / Layer 1 分離可視化＋標高プロファイル
- `walk_3d_trajectory.json`: 3D A\* 経路＋フレーム毎の `x, y, z, yaw_deg, slope_deg, surface_type, layer, gait_phase, torso_pitch_deg, knee_lift_mul, stride_mul`
- `showcase_<theme>_4view.png`: Spark 4 アングル確認シート（`--spark-dir` 指定時）
- `decoupled_outputs_triptych.png`: 分離出力 Albedo / Depth / Surface Normal マップ（`--export-maps` 指定時）

### 3. リグ済み 3DGS キャラクターの立体ウォークスルー動画（`render_stage_walk.py`）
```bash
python skills/gs-stage/scripts/render_stage_walk.py \
  --map-ply work/stage_ruins/map_forest_ruins.ply \
  --trajectory work/stage_ruins/walk_3d_trajectory.json \
  --char-v1 work/hood-01/05_rig/v1 \
  --out-dir work/stage_ruins \
  --spark-dir <spark_demo>
```
- 統合マップを Spark の `static` レイヤーとして 1 回だけ GPU に載せ、キャラのみ毎フレーム差し替えるため、64 フレームの立体交差ウォークスルーを約 25〜40 秒で生成できる。

---

## 品質チェックポイント
- **浮き・めり込み確認**: `showcase_<theme>_4view.png` で垂れツタ・石門・井戸・岩の接地を必ず目視確認する。
- **立体交差の検証**: 2 層以上の構造では `nav_multilayer_3d_preview.png` の `Overlap cells > 0` と標高プロファイル `Z(s)` にワープ（垂直ジャンプ）がないことを確認する。
- **目の疲れ・彩度バランス**: テーマの環境光と点光源は過飽和を避け、プロップ群（`forest_pack` 等）の色調と自然に馴染ませる。
