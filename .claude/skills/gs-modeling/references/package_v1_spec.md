# 3DGS 標準アセットパッケージ契約（v1）＆ トラブルシューティング

## 1. 標準パッケージディレクトリ構造（`<asset_id>/v1/`）
自動QAを通過した確定版のみを不変バージョンとして凍結する。下流（FDM印刷ルート / アニメーションルート）はこのパッケージのみを入力とする。

```text
<asset_id>/v1/
├── meta.json               # 単体メタデータ・対応パイプライン宣言・自動QAスコア
├── preview.png             # Spark 3DGS 正面サムネイル
├── preview_rig.png         # スケルトン・プロシージャルケージ重畳デバッグプレビュー
├── master.ply              # SSoT: 正規化・平滑化・ドールアイ統合済み 3DGS (Z-up, -Y-forward, height=1.0, SH0, logit/log)
├── rig_skeleton.json       # ボーン階層・統一ローカル軸 ([1,0,0,0])・可動域リミッター (limits_deg)
├── rig_cages.obj           # 3D連結成分ごとに自動生成した超ローポリケージ（マージン付き）
├── control_points.json     # 変身・変形・適応Smooth制御ポイント
└── exports/                # 最終成果物のみ（中間ファイル禁止: .mp4, .3mf 等）
```

## 2. ローカル実行環境リファレンス（環境変数ベース）
- **Python**: `python` 3.10+（`numpy`, `scipy`, `pillow`, `imageio` 等。`pip install -e ".[all]"` で導入）
- **ComfyUI**: 環境変数 `GS_COMFYUI_DIR` 配下の `main.py`、API エンドポイント `GS_COMFYUI_URL`（既定 `http://127.0.0.1:8188`、任意で `--extra-model-paths-config` を指定）
- **Blender**: 環境変数 `GS_BLENDER_BIN`（既定は `PATH` 上の `blender`、`4.2+` / `5.x` 推奨）
- **Spark Renderer**: リポジトリ内 `renderer/spark_demo/`（`capture.mjs`, `capture.html`, `capture.js`, `npm install && npm run sync-vendor` で `vendor/` を配置。GPU ANGLE D3D11 + `SplatSkinning` による `anim` モード対応、`SPARK_SOFTWARE=1` でソフトウェア描画へフォールバック可能）
- **Spark座標変換**: Blender（`Z-up, -Y-forward`）→ Spark 入力へ `R_x(+90°)` を適用（`capture.js` の `root.rotation.x = Math.PI` と組み合わさって直立正面になる）。

## 3. 実証済みトラブルシューティング（既知の地雷と回避策）

| 症状 | 原因 | 回避策 |
|---|---|---|
| 両足の間に背景の板が生成される | 足元の影で外周と分断され `floodFill` が入らない | 外周 `floodFill` に加え、背景固有色の内部検出マスクを併用 |
| ComfyUI が `custom_nodes` 不在で起動失敗 | `--base-directory` 配下に空ディレクトリが無い | 起動前に `custom_nodes`, `models`, `user`, `input`, `output`, `temp` を作成 |
| 生成モデルが上下逆＋横向き＋約28°後傾 | TripoSplat の座標系（`Up=-Y, Front=+X`）と仰角バイアス | `step1_normalize.py` で直交回転＋足元〜頭部の YZ 重心傾きによるピッチ直立補正 |
| Blender CLI で `--front-axis -Y` がエラー | 先頭ハイフンが `argparse` にオプション誤認される | `--front-axis=-Y` のように `=` 連結で渡す |
| Node.js `JSON.parse` が先頭文字で失敗 | PowerShell 5.1 `Set-Content -Encoding UTF8` が BOM を付与 | `[System.IO.File]::WriteAllText(..., UTF8Encoding($false))` を使用 |
| ドールアイが三白眼で怖く見える | 虹彩が小さい真円＆前面点密度不足 | 前面 `72°` 高密度キャップ（約4,500点/目）＋縦長楕円虹彩（`37°×45°`）＋`smoothstep`＋2点キャッチライト |
| 腕を振ると指先がちぎれスカートが飛ぶ | ユークリッド距離（カプセル等）の決め打ち | `connected_components`（kNN半径 `0.016`）で腕連結成分を厳密抽出 |
| 肘が2箇所（クランク状）に見える | ①手首検出の下限が低すぎ前腕中央を誤認 ②フリル袖のフチに肩がズレ上腕がたわむ | ①手首スキャンを `s ∈ [0.72, 0.88]` に限定 ②前腕 PCA 直線から袖内側（`±0.078`）へ肩を逆算＋体幹Yクランプ＋左右対称化＋C1連続袖ウェイト |
| 前腕がねじれて歪む | 肘曲げブレンドと前腕ツイストが同じ `s` 区間で干渉 | 肘曲げを `s_el ± 0.065` に絞り、前腕ツイストは `s_el + 0.05 → s_wr`・上限 `±58°` に分離 |
| 閉眼時に白い滲み・斜め凹みが出る | 眼窩奥の眼球表面の色書き換え＋白目フチ混入 | 顔面2次曲面フィットの独立まぶたシェル（眼間距離 `D_eye` に比例した相対サイズ、前髪ライン以下へ上端クリップ）＋閉眼側眼球の奥退避 |
| 瞬き時に上から謎の線が降りてくる | 大きめシェルの最上端（おでこ／前髪）からまつ毛線を開始 | まつ毛の降り始めを本来の上まつ毛位置に固定し、`blink` 立ち上がりでソフトフェードイン |
| 数値QAはPASSなのに顔・手・体格が破綻している | ①入力正面図の解像度不足（顔<120px）②実装者と検品者が同一で相対評価バイアスが働く ③ちぎれ率のみの指標ハック | ①Gate 0（`gate0_check_input.py`）で入力解像度不足を事前 Hard Stop ②独立 `qa_reviewer` サブエージェントによる Defect-First 検品（Maker-Checker分離）③肩幅膨張比・手首高さ誤差などの体積保存ゲート追加 |
