---
name: gs-modeling
description: >-
  Modular local 3D Gaussian Splatting (3DGS) character asset & rigging pipeline in five stages:
  input preprocessing, ComfyUI / TripoSplat generation with multi-seed comparison, procedural body
  cleanup (posture, surfaces, detail), mandatory dynamic eyes/eyelids and procedural mouth, and a
  mandatory v1 rigged package with head-turn check and a short QA video. Optional props, performance
  videos, and handoff to gs-to-sprite / gs-to-mesh. Use when making a 3DGS .ply character from a 2D image.
---

# 3DGS Character Asset & Rig Pipeline（5 段階）

## スキルの目的
**デザイン画に準拠した綺麗な 3D ガウススプラット（`.ply`）を作ること**、および **アニメーションで使いやすいリグ・動的ドールアイ（瞬き・ウインク・視線移動）・動的プロシージャル口（開閉・あいうえおリップシンク）の仕組みを組み込むこと** が本スキルの目的である。
ガウススプラット生成および各修正の節目では、個別の数値や一部パーツだけでなく、**デザイン画の印象・バランス・クリーンさを総合的に満たせているか（全体確認）** を必ず確認する。

---


## 核心ルール（最小限の必須事項）
1. **Python 処理・Blender は確認レンダリング専用**: 座標変換・切り出し・マージ・変形は Python（NumPy/SciPy）で行い、Blender GUI 統合は行わない。
2. **ディスク空き容量の確保**: 作業前に作業ボリュームの空き容量を確認し、10 GB 未満の場合は十分な空きがある別ボリュームへ作業ディレクトリと `TEMP`/`TMP` を逃がす。
3. **UTF-8（BOM なし）の徹底**: Node.js（`capture.mjs`）が読む JSON は必ず BOM なし UTF-8 で書き出す（Windows PowerShell 5.1 等の BOM 付与に注意）。
4. **データ駆動（固定値の禁止）**: 顔の輪郭・目の位置・ボーン位置・本数などを特定キャラの固定値で直書きせず、3DGS 点群の実測値（断面重心・ランドマーク・相対比率）から算出する。
5. **デザイン画との総合一致確認 ＆ プロシージャル優先の整理原則（必須）**:
   - **姿勢 → 大きな面・輪郭 → 細部**: 姿勢と面を順に確定してから、その面に依存する修正へ進む。上流を作り直した場合は影響範囲の下流を再確認する。
   - **余計な再生成・継ぎ接ぎ結合をしない**: TripoSplat 等のモデル生成が細部（目・口・襟・柄・小物等）で粗くなるのは前提とする。粗い箇所を別モデル再生成の継ぎ接ぎで誤魔化さず、**3DGS の編集しやすさを活かして元デザイン画（`front_raw.png`）を基準にプロシージャルに直す**（汚れた古いスプラットを中途半端に残して上塗りせず、対象領域を更地化してから単一 3D 曲面へ高密度マイクロスプラットで再構成する）。
   - **数値は異常検知の最低ライン**: `verify_gate.py` の数値（縦横比・浮遊点率・ボーン偏差など）は破綻や見切れを弾くための安全網であり、数値を合わせるために顔の輪郭や立体バランスを壊す非線形ワープを行ってはならない。
   - **全体確認を抜かさない**: 局所調整の過程で一時的に局所最適へ寄るのは許容しつつも、節目ごとに必ず **元デザイン画と全体像（アップ＋斜め／全身）を同一基準で確認** し、「印象・バランス・アップ時のクリーンさ（まだら・二重・ズレなし）」を総合的に満たしてから動画化・提出へ進む。
6. **調整ループの軽量化 ＆ QA の実行タイミング分離（過剰パッチ防止）**:
   - **調整中（Inner Loop）**: レンダーと計測結果は作業フォルダーに保存する。形状の判断に目視が必要なときは、該当部位と隣接部を同じ視点・尺度で並べた比較画像を確認する。途中経過の共有は原則テキストにし、全アングルの撮り直しと `verify_gate.py`・独立 QA は確定時にまとめる。
   - **効率化（[efficiency.md](./references/efficiency.md)）**: 1 回の試行は [check_candidate.py](./scripts/check_candidate.py) 1 回（数値 → 通ったときだけレンダー → 1 行の要約）。数値で落ちた候補は見ない。目視の確認はサブエージェントに任せて文字で受け取り、自分で見るのは判断の場面だけ、[make_sheet.py](./scripts/make_sheet.py) の縮小シート 1 枚で見る。段階ごとに確定ファイルと `qa/stage_<n>_handoff.md` で引き継ぐ。
   - **確定時（Milestone Check）**: 仕上がった段階で **1 回だけ** `verify_gate.py`（数値・幾何の機械チェック）と、元デザイン画を併記した独立ブラインド QA を実行する。
   - **ユーザーへの画像共有**: 採用候補がまとまったとき、方針の判断が必要なとき、ユーザーが求めたときに、判断に必要な角度を比較シートにまとめて見せる。全サイズのレンダーや他の角度は作業フォルダーに残し、必要に応じてリンクを添える。
   - **独立 QA の原則**: 修正経緯を伏せたブラインド状態で、「元デザイン画と 1:1 で並べてユーザーに見せられる品質か（PASS / FAIL）」と**デザイン画にないゴミ・破綻・致命傷のみ**を簡潔に判定する（※些末な指摘を無理やり絞り出す運用は禁止）。
   - **QA 前セルフチェック**: 独立 QA を呼ぶ前に、[quality_checklist.md](./references/quality_checklist.md) のゲート別チェック項目を「済 / 未」で書き出す。未が残っているうちは QA を呼ばない。独立 QA は同ファイルの依頼文テンプレートと固定の撮影セットで依頼し、判定のぶれを減らす。
   - **早期停止（最大 3 回）**: 問題に ID を振って `qa/attempt_log.md` に試行を記録する（言い換えでも同じ部位・同じ原因なら同じ ID）。同じ ID が 3 回で収束しない場合は即座に停止し、現状画像を添えてユーザーへ方針確認する。


---

## 5 段階の流れ

| 段階 | 目的 | 確定ファイル | ゲート | 詳細 |
| :--- | :--- | :--- | :--- | :--- |
| 1. 入力の前処理 | 背景透過・3D 化適性・原画の目印の計測 | `01_input/front_rgba.png`, `landmarks.json` | Gate 0 | [stage1_input.md](./references/stage1_input.md) |
| 2. 生成 | 2〜4 シードを生成・正規化し、比べて 1 つ選ぶ | `02_gen/chosen.ply`, `choice.md` | 数値（点数・NaN・高さ） | [stage2_generate.md](./references/stage2_generate.md) |
| 3. 全体の整理 | 姿勢 → 大きな面・輪郭 → 静止ディテール。最後にリグを付けやすい姿勢かを確認 | `03_body/body.ply` | Gate 1 | [stage3_body.md](./references/stage3_body.md) |
| 4. 顔の調整（必須） | 顔ディテールの整理 → 動く目 → 動く口 | `04_face/head_base.ply`, `eyes/`, `mouth/` | Gate 2 | [stage4_face.md](./references/stage4_face.md) |
| 5. リグ付け（必須） | ボーン・ウェイト・`v1/` パッケージ・首振り ±18°・短い確認動画 | `05_rig/v1/`, `check.mp4` | Gate 3・Gate 4 | [stage5_rig.md](./references/stage5_rig.md) |

- **段階を飛ばさない。** 段階 4・5 はすべてのキャラで行う（静止の目・口、リグなしで終えない）。
- **各段階は前の段階の確定ファイルだけを読む。** 確定ファイルは上書きせず、やり直すときは影響を受ける段階から再開する。段階を終えたら `qa/stage_<n>_handoff.md` を書く（[efficiency.md](./references/efficiency.md) 4 章）。
- **各段階の順序は「数値の検査 → QA 前セルフチェック → 独立 QA」。** 数値で落ちたら目視に進まない。数値が通っても目視で落ちることはあり、確定時の固定撮影セット（真横 90° を含む）は必ず見る。
- **該当するキャラ条件があれば追加手順を読む**: 顔のアクセサリー / 長い髪・マント / 背面資料あり / 細密キャラ（262k 点）→ [character_conditions.md](./references/character_conditions.md)
- 頭部だけを別生成して移植する方法は採らない（継ぎ目・色ムラで破綻したため廃止）。顔の解像度が足りなければ段階 1 で原画を差し替える。

### 作業フォルダ

キャラ固有の設定は [character_config.template.json](./resources/character_config.template.json) を `character_config.json` としてコピーして使う。環境パスと `v1/` パッケージの仕様は [package_v1_spec.md](./references/package_v1_spec.md)。

```
<work>/
  character_config.json
  01_input/   front_raw.png, front_rgba.png, landmarks.json
  02_gen/     seeds/seed_<n>.ply, seeds_sheet.png, chosen.ply, choice.md
  03_body/    body.ply（確定）, work/（途中の候補）
  04_face/    head_base.ply, eyes/, mouth/, face_states.json
  05_rig/     v1/（パッケージ）, check.mp4
  qa/         views/<段階>/, attempt_log.md, <段階>_qa.md, gate_<n>.json, stage_<n>_handoff.md
```

---

## 検証ゲート一覧（Layer 1 機械チェック ＋ Layer 2 デザイン画突合）

各ゲートは **`verify_gate.py`（Layer 1：数値・幾何・比率・浮遊点・オーバーレイ自動生成）** と **元デザイン画付き独立ブラインド QA（Layer 2：美観・掃除漏れ・破綻判定）** の 2 層で完結させる。独立 QA の前に [quality_checklist.md](./references/quality_checklist.md) のセルフチェックをすべて「済」にする。

| ゲート（段階） | Layer 1: 機械チェック | Layer 2: 視覚・デザイン画突合（独立ブラインド QA） |
| :--- | :--- | :--- |
| **Gate 0**（段階 1） | 解像度・背景透過・顔幅（`gate0_check_input.py`） | 透過漏れ・明色穴あき・ポーズ適性 |
| **Gate 1**（段階 3） | 点数 / NaN / 高さ `1.0` / 3D 浮遊点 / 1:1 デザイン画縦横比（`<= ±10%`）＋ 50/50 オーバーレイ / **真横の体幹の傾き ±8°・靴底 ±3°** | 頭身・輪郭の伸び/潰れ・肩/襟/背面の浮遊ゴミ・破綻、真横の姿勢 |
| **Gate 2**（段階 4） | バスト 1:1 縦横比＋ 50/50 オーバーレイ / 3D 浮遊点 / Before-After 回帰シート | 顔全体・鼻筋・頬・目の下・顔のアクセサリー・首元の掃除漏れゼロ、二重まぶた・白目漏れ |
| **Gate 3**（段階 5） | 全ボーン内包（`<= 0.004`）・断面重心のずれ・左右対称・X 線 3 面図 ＋ **首振り ±18° の顔拡大** | 関節位置の自然さ・ポーズ破綻、首振りで目・口がずれない |
| **Gate 4**（段階 5） | 先行ゲート通過・動作尺（`>= 2.0s/動作`）・上端見切れ全フレーム走査・Frame 0 vs デザイン画 | **`.mp4` 本体を直接確認**（揺れ・ちらつき・まぶた破綻・掃除残しゼロ） |

---

## 用途オプション（用途に応じて選ぶ・強制しない）

詳細は [options.md](./references/options.md)。

- **小物の追加**: リュック・鞘・手持ち武器などを単体で生成し、親ボーンへ剛体で取り付ける（本体は小物なしで段階 1〜5 を通す）。
- **演技ムービー・納品用の動画**: 凍結した `v1/` だけから作り、Gate 4 と `.mp4` 本体の独立 QA を通してから提出する。
- **下流スキルへの受け渡し**: gs-to-sprite（2D スプライト）、gs-to-mesh（印刷用メッシュ）へ `v1/` を渡す。
