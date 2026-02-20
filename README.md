# Rcc_PIVの実装

### ディレクトリ構成
```txt
Rcc_PIV/
├── data/               # 学習用データセット置き場
│   ├── images/         # PIV画像 (img1, img2)
│   └── flows/          # 正解データ (ground truth flow)
├── models/             # モデル定義のコード
│   ├── __init__.py
│   ├── uraft.py        # メインモデル (URAFTクラス)
│   ├── extractor.py    # 特徴抽出器 (Encoder)
│   └── update.py       # GRU更新ブロック
├── utils/              # 便利関数
│   ├── flow_viz.py     # 可視化用
│   └── dataset.py      # データローダー
├── train.py            # 学習実行スクリプト
└── inference.py        # 推論実行スクリプト
```




