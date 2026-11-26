# 段階 2: 生成（2〜4 シードを比べて選ぶ）

- 入力: `01_input/front_rgba.png` → 出力（確定）: `02_gen/chosen.ply`, `02_gen/choice.md`（選んだ理由）
- 途中: `02_gen/seeds/seed_<n>.ply`, `02_gen/seeds_sheet.png`
- 順序: 数値の検査（点数・NaN・高さ 1.0）→ シード比較シートで選ぶ → 候補が拮抗したときだけユーザーに相談

## 手順

1. ComfyUI の `--base-directory` 配下に `custom_nodes`, `models`, `user`, `input`, `output`, `temp` を作ってから起動し、[step0_generate_ply.py](../scripts/step0_generate_ply.py) で 2〜4 シードを生成する（標準 `65,536` 点。細密・高頭身キャラは `262,144` 点。[character_conditions.md](./character_conditions.md)）。
2. 各シードを [step0_inspect_ply.py](../scripts/step0_inspect_ply.py) → [step1_normalize.py](../scripts/step1_normalize.py) で正規化する（直立・Z-up・-Y 正面・高さ `1.0`・`nan_or_inf == 0`）。数値で落ちたシードはレンダーしない。
3. 残ったシードを固定の撮影セット（全身 `0° / 90° / 180°`、顔 `0° / +60°`）で撮り、[make_sheet.py](../scripts/make_sheet.py) で `seeds_sheet.png` 1 枚にまとめて比べる。
4. **細部ではなく、姿勢・正面の比率・顔と頭の厚み・背面の破綻の少なさで選ぶ。** 細部はプロシージャルに直せるが、姿勢・厚み・比率の崩れは段階 3 の修正回数を大きく増やす。[check_candidate.py](../scripts/check_candidate.py) の真横の傾き・浮遊点も判断材料にする。
5. 選んだシードを `chosen.ply` として保存し、`choice.md` に候補ごとの一言と選んだ理由を書く。

## 確定の条件

- 数値（点数・NaN・高さ）が通り、比較シートで選んだ理由が記録されている。
- 全シードで同じ破綻（顔が平たい・背面の板など）が出る場合は、原画側の問題を疑い段階 1 に戻る。
