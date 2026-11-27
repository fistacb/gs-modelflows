# `mode: static_prop` — テレイン・小物アセット生成ガイド

木・岩・街灯・宝箱・建物・家具などの**静止オブジェクト（ボーンなし・顔なし）**を、キャラクターと同じ `gs-modeling` の Stage 1〜3 品質で作って凍結するための軽量モードです。

---

## フロー（Stage 1〜3 → `v1_prop/` 凍結）

Stage 4（動的目・口）および Stage 5（22 ボーンリグ・動画 QA）はスキップします。

1. **Stage 1（入力の前処理）**: 背景透過（`step0_prep_image.py` / `step0_matte_comfy.py`）。正面または 3/4 の単体画像を用意。
2. **Stage 2（TripoSplat 生成・シード比較）**: 2〜3 シード生成し `step1_normalize.py` で正規化して破綻の少ない 1 つを選定。
3. **Stage 3（姿勢・浮遊ゴミ除去・全体確認）**: 傾き・底面接地を整え、`remove_detached.py` で浮遊ゴミ・エッジのヒゲを除去。
4. **凍結パッケージ化（`build_static_prop.py`）**:
   - 底面中央ピボット（`Z = 0`、`X = Y = 0`）と実寸スケーリング（`--target-height-m`）を適用し、`v1_prop/` へ出力する。

---

## 実行コマンド例

```bash
python skills/gs-modeling/scripts/build_static_prop.py \
  --ply work/chest-01/03_body/prop.ply \
  --out-dir work/chest-01/v1_prop \
  --name treasure_chest \
  --category chest \
  --target-height-m 0.75 \
  --clean \
  --render-qa
```

### 出力構成（`v1_prop/`）
- `prop.ply`: `Z-up / -Y-forward / 底面中央ピボット / 実寸（m）` 済み 3DGS
- `meta.json`: 実寸外寸（`dimensions_m`）、`bbox_min_m` / `bbox_max_m`、`footprint_radius_m`、`collision_cylinder`（`gs-stage` の `anchor` 配置・当たり判定に直結）
- `preview_8view.png`: 8 方向ターンアラウンド確認シート（`--render-qa` 指定時）

---

## 下流スキルへの受け渡し

- **2D / HD-2D スプライト化（`gs-to-sprite`）**:
  ```bash
  python skills/gs-to-sprite/scripts/render_hd2d_sprites.py \
    --ply work/chest-01/v1_prop/prop.ply \
    --out-dir work/chest-01/sprites \
    --size 80 --colors 48 --export-maps
  ```
- **3D プリント用 STL / GLB 化（`gs-to-mesh`）**:
  ```bash
  python skills/gs-to-mesh/scripts/extract_density_mesh.py \
    --ply work/chest-01/v1_prop/prop.ply \
    --out work/chest-01/06_mesh/prop_body.npz --opening 1
  ```
- **背景・ステージ配置（`gs-stage`）**: `v1_prop/meta.json` の `footprint_radius_m` と `pivot = bottom_center` を参照し、床面アンカーへそのまま接地配置できる。
