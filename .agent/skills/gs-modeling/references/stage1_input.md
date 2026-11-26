# 段階 1: 入力の前処理（Gate 0）

- 入力: 原画 → 出力（確定）: `01_input/front_raw.png`, `01_input/front_rgba.png`, `01_input/landmarks.json`
- 順序: 数値の検査（`gate0_check_input.py`）→ 自分の確認（透過結果 1 枚）→ 3D 化に向かない原画のときだけユーザーに相談

## 原画の条件

- **単一画像前提（ローカル TripoSplat）**: TripoSplat は単一画像専用。4 面シートより **高解像度な単体正面図 1 枚** を優先する（4 面シートしかない場合は [step0_prep_image.py](../scripts/step0_prep_image.py) で正面を切り出す）。背面・側面の資料は段階 3 の参考に回す（[character_conditions.md](./character_conditions.md)）。
- **3D 化適性（横顔・首元の破綻防止）**:
  - **塗り・輪郭線**: 太い黒アウトライン（特に顎・耳・首周り）は横顔の黒壁・えぐれになりやすい。黒線の弱い、立体陰影のある絵が向く。
  - **表情**: 全開の歯見せ笑顔は横顔が平たく破綻しやすい。閉じた口か穏やかな微笑みが向く。
  - **ポーズ**: 段階 5 でリグを付けるため、**75°〜85° のリラックス T ポーズ、手のひらを正面に向けた状態** が望ましい（指の分離と前腕ツイストが安定する）。
- **顔の解像度**: 顔幅の目安は `>= 120 px`。足りない場合は高解像度の単体図へ差し替える。**頭部だけを別生成して移植する方法は採らない**（継ぎ目・色ムラで破綻した実績があるため廃止）。

## 手順

1. [step0_prep_image.py](../scripts/step0_prep_image.py) で背景を透過し `front_rgba.png` を作る。背景がキャラの色に近い・背景にグラデーションや影がある・白い服に白背景・細い髪などで色の切り抜きが崩れるときは、[step0_matte_comfy.py](../scripts/step0_matte_comfy.py)（ComfyUI の BiRefNet）を使う。外周の背景残り・足の間など閉じた空間の背景残り・明るい色のパーツの穴あきがないかを透過結果 1 枚で確認する（腕と胴の間など背景色の閉じた穴は、大きいもの（`--min-hole`、既定 800 px）は常に、小さいもの（`--min-outlined-hole` 以上）は周りが暗い輪郭線のとき（`--outline-L`）だけ透過する。周りが明るい小さな穴はハイライト・宝石・白い服とみなして残す。透過した穴は実行時に一覧が出る）。
2. [gate0_check_input.py](../scripts/gate0_check_input.py) で解像度・透過・顔幅・ポーズを検査する。FAIL なら段階 2 に進まない。
3. 原画の目印（目・顎・肩・腰・裾・足首・髪先など、そのキャラで位置が読める部位）の画素位置を読み、[measure_landmarks.py](../scripts/measure_landmarks.py) で高さ（靴底 0・頭頂 1）・輪郭の幅・`D_eye` を `landmarks.json` に残す（例: `--mark eye_l=402,286 --mark eye_r=470,286 --mark chin=330`）。段階 3 のランドマーク高さの比較に使う。
4. [character_config.template.json](../resources/character_config.template.json) を作業フォルダへ `character_config.json` としてコピーし、キャラ条件（長い髪・マント・小物など）を書く。

## 確定の条件

- `gate0_check_input.py` が PASS、透過漏れ・穴がない、`landmarks.json` がある。
