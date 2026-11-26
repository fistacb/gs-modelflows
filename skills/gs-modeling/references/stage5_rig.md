# 段階 5: リグ付け（Gate 3・Gate 4・必須）

- 入力: 段階 4 の出力 → 出力（確定）: `05_rig/v1/`（パッケージ）, `05_rig/check.mp4`
- **すべてのキャラで必ず行う。** ボーン・ウェイト・`v1/` パッケージ・首振り ±18° の確認・短い確認動画までを 1 段階とする。
- 順序: 数値の検査（`verify_gate.py --gate 3`・`--gate 4`）→ 自分の確認（X 線 3 面図、動画本体）→ 独立 QA → 完了時に動画で報告。
- 調整中に撮るのはポーズ 1〜2 枚。

## 1. ボーンとウェイト・`v1/` パッケージ

- [step7_build_v1_package.py](../scripts/step7_build_v1_package.py) と [harmonic_dqs.py](../scripts/harmonic_dqs.py)（3D kNN 調和ラプラシアン＋DQS＋前腕ツイスト＋動的スプラット補間）で骨格・ウェイト・`v1/` パッケージを作る。パッケージの構成は [package_v1_spec.md](./package_v1_spec.md)。
- 骨格は点群の実測（断面重心・ランドマーク）から求め、キャラ固有の固定値を書かない。
- **頭蓋骨の剛体化**: `compute_harmonic_skin_weights` で顎より上（顔・目・口・髪・帽子・顔のアクセサリー）を `head = 1.0` の剛体にする。頭から 3D でつながる背面・側面の長い髪も `head` に含める（[character_conditions.md](./character_conditions.md)）。
- 腕の大振り・上体の回転の符号とジンバルの注意は [package_v1_spec.md](./package_v1_spec.md) のトラブルシューティング。

## 2. 首振り ±18° の確認（必須）

- 段階 4 の目・まぶた・口を `head` に追従させた状態で、首を左右 ±18° 振った顔の拡大を撮る。目の輪郭線・眼球・口が顔面からずれない・浮かないことを確かめる。
- ずれる場合は、口の `w_mouth_anchor`、まぶたシェルの追従先、`head` に入っていない顔の部分を先に疑う。

## 3. 確定時のリグ検査（Gate 3）

- `verify_gate.py --gate 3`（全ボーンの内包 `<= 0.004`・断面重心のずれ・左右対称・X 線 3 面図）。3 面図は [render_rig_preview_blender.py](../scripts/render_rig_preview_blender.py)。
- 自分で X 線 3 面図とポーズ 1〜2 枚を見て、関節位置が自然か、ポーズで破綻しないかを確かめる。

## 4. 短い確認動画（Gate 4）

- [render_standard_qa_video.py](../scripts/render_standard_qa_video.py) で標準の確認動画（顔: 瞬き・視線・首振り・ウインク・口パク／上半身: 腕／全身: 歩く・走る・跳ぶ）を作り `check.mp4` とする。
- Spark GPU の `SplatSkinning`（`capture.mjs anim`）を使い、毎フレームの全身 PLY 書き出しはしない。全身 PLY とスキンウェイト（`pack_spark_skin_weights` → `body_skin.bin`）を最初に 1 回だけ読み込み、フレームごとにボーン姿勢（`fk_to_spark_bones`）だけを渡す。表情パートは目・まぶた・口だけの軽い PLY を `extraPly` で差し替え、全身のパートは静止の目・まぶた・口 PLY を `extraTransforms` で `head` に剛体追従させる。
- `verify_gate.py --gate 4 --video 05_rig/check.mp4 --num-segments <N> --min-seg-sec 2.0 --out-dir qa`（1 動作 `>= 2.0` 秒・上端の見切れ 0・Frame 0 と原画の比較）。
- 独立 QA は静止画だけでなく **`.mp4` 本体** で確認する（揺れ・ちらつき・まぶたの破綻・掃除残し）。

## 5. 完了

- Gate 3・Gate 4・独立 QA が通ったら `v1/` を凍結し、`qa/stage_5_handoff.md` を書き、ユーザーへ動画で報告する。
- 演技ムービー・納品用の動画、下流スキルへの受け渡しは [options.md](./options.md)。
