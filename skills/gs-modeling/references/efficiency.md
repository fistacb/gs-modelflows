# 作業の効率化（品質を落とさずに時間と消費トークンを減らす）

どれも「確認を減らす」ためではなく、「同じ確認を安く行う」ためのもの。各段階の確定時の固定撮影セット（真横 90° を含む）、`verify_gate.py`、QA 前セルフチェック、独立 QA、3 回で止めるルールは省略しない。

## 1. 1 回の試行 = 1 回のスクリプト呼び出し

調整中の候補は [check_candidate.py](../scripts/check_candidate.py) でまとめて調べる。

```
python scripts/check_candidate.py --cand 03_body/work/try3.ply --ref 02_gen/chosen.ply \
    --protect qa/protect_boxes.json --out qa/views/stage3/try3 --render
```

- 数値（点数・NaN・高さ・3D 浮遊点・真横の体幹の傾き・靴底の傾き・過大スプラット・保護範囲の移動量・動いた範囲）を先に測り、**数値が通ったときだけ**レンダーする。
- 会話に返るのは 1 行の要約だけ。詳しい値は `<out>.json` に残る。
- 変形は [splat_ops.py](../scripts/splat_ops.py) の `transform_gaussians` を使う（位置と一緒にガウスの向き・広がりを更新し、折り返す変形は `ValueError` で止まる）。キャラごとに変形器を書き直さない。

### 標準の道具（キャラごとに書き直さない）

| 道具 | 用途 |
| :--- | :--- |
| [render_views.py](../scripts/render_views.py) | 固定撮影セット（全身 `000 / p035 / m035 / p090 / p180`、バスト）。`--only full:90` で 1 角度だけ、`--ref <前の PLY>` で前後を同じカメラ・同じ尺度で撮る。 |
| [correct_posture.py](../scripts/correct_posture.py) | 真横の前傾・後傾を線形シアーで取る（靴底は動かさず、首から上は剛体で移動）。報告に体幹の傾きの前後と足の移動量が出る。 |
| [align_rows.py](../scripts/align_rows.py) | 目印の高さを原画に合わせる（高さ方向だけの単調な伸縮）。 |
| [measure_landmarks.py](../scripts/measure_landmarks.py) | 原画・正面レンダーの目印の高さ、輪郭の幅、`D_eye` を測る。`--pairs-with` で `align_rows.py` 用の対応表を書く。 |
| [remove_detached.py](../scripts/remove_detached.py) | 体につながっていない浮遊塊を除く。報告の塊（中心・大きさ）を見てから採用する。 |

```
python scripts/correct_posture.py --ply 02_gen/chosen.ply --out 03_body/work/posture.ply
python scripts/render_views.py --ply 03_body/work/posture.ply --out qa/views/stage3/posture --only full:90 --ref 02_gen/chosen.ply
```

## 2. 数値の検査が先、目視は後

- 数値で落ちた候補はレンダーしない・見ない。
- 目視で見つかった問題のうち数値にできるものは、`check_candidate.py` と `verify_gate.py` に足す。例: 真横 90° の体幹の傾き（`verify_gate.py --gate 1` でも検査。既定 ±8°）。
- 傾きのしきい値を変えてよいのは、デザイン上の理由があるときだけ（大きなマント・尻尾・後ろに広がる髪・手に持った杖などで、中央値の線が引っ張られる場合）。`character_config.json` の `qa_thresholds.max_trunk_lean_deg` を変え、理由を同じファイルか `qa/attempt_log.md` に書く。実測例: 人型で直したもの +1°前後、直す前の sakura +40°、snowdrop の見逃し +21°、鳥の魔法使い +16°、杖を持つカエル −11°。

## 3. 画像はメインの会話に入れない

- **目視の確認はサブエージェントに任せ、結果は文字で受け取る。** 依頼文の例:

  ```
  次の画像を見て、文字だけで答えてください。画像を返したり説明を長く書いたりしないでください。
  - 原画: <front_raw.png のパス>
  - 候補: <レンダーのパス>（角度: 90°、確認したい部位: 体幹の傾きと靴底）
  判定項目:
  1. <項目>
  2. <項目>
  返答の形式（項目ごとに 1 行）:
  <番号>. PASS か FAIL — FAIL のときは、どこがどう悪いかを 1 行で
  最後の行: 総合 PASS / FAIL
  ```

- 独立 QA も同じく、結果は文字で受け取る（依頼文は [quality_checklist.md](./quality_checklist.md) のテンプレート）。
- **自分で画像を見るのは判断の場面だけ**（シードの選択、候補 2 つの比較、3 回で止まったときの方針決め、確定前の最終確認）。そのときは [make_sheet.py](../scripts/make_sheet.py) で 1 枚の縮小シート（長辺 1024 px 程度）にまとめてから見る。

  ```
  python scripts/make_sheet.py --out qa/views/stage3/compare.png before_full_p090.png:"before 90" after_full_p090.png:"after 90"
  ```

  `render_views.py` の出力名は `<out>_full_<角度>.png`（角度は `000`, `p035`, `m035`, `p090`, `p180`）。同じパスに上書きした画像は表示が古いまま残ることがあるので、撮り直しは新しいファイル名にする。

- ユーザーへの画像報告は、確定時・3 回で止まったとき・方針の判断が要るときだけ。

## 4. 段階ごとに区切って引き継ぐ

- 各段階は前の段階の**確定ファイル**だけを読む（フォルダ構成は [SKILL.md](../SKILL.md) の「作業フォルダ」）。
- 段階を終えたら `qa/stage_<n>_handoff.md` を書く。次の段階はこれと確定ファイルだけで始められるようにし、長い作業はサブエージェントに 1 段階ずつ渡してよい（その場合も確定時の verify_gate・セルフチェック・独立 QA はその段階の中で済ませる）。

  ```markdown
  # 段階 <n> 引き継ぎ
  - 確定ファイル: <パス>（作成日時）
  - 数値: verify_gate gate <n> = PASS（qa/gate_<n>.json）、傾き <x>°、靴底 <y>°、浮遊点 <z>%
  - セルフチェック: 全項目 済（qa/stage_<n>_qa.md）
  - 独立 QA: PASS（日時、指摘の要約 1 行）
  - 未解決・持ち越し: <問題 ID と内容>（qa/attempt_log.md の ID）
  - 次の段階への注意: <保護範囲、触ってはいけない部位など>
  ```

## 5. 効果の測り方

新しい手順で 1 体を通したら、所要時間・レンダー回数・やり直し回数・QA 通過に加えて、消費トークン（キャッシュ読み込み量・呼び出し回数・会話に読み込んだ画像の枚数）を今と比べる。品質が下がった変更は元に戻す。
