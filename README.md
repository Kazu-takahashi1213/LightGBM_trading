This project builds upon concepts learned in the intensive program organized by the **Matsuo Laboratory at the University of Tokyo**, which focuses on:

> "Development of Machine Learning Models for Financial Market Trading"

I successfully completed the program in **November 2024**, having passed the final evaluation.  
Below is the official certificate of completion:


![修了証 pdf](https://github.com/user-attachments/assets/71ee6305-2c68-4191-88b6-9fa2b82f59ff)

<br>

````markdown
# Financial Trading Algorithm

このプロジェクトは、**金融市場分析**と**機械学習**を組み合わせた自動売買アルゴリズムの開発を目的としています。
特に状態空間モデル（カルマンフィルタ）によるレジーム推定を中核とし、以下のモジュールで構成されています。

---

## 📂 ディレクトリ構成

```plaintext
Financial_trading_algolythm/
├── .gitignore                # Git管理外ファイル定義
├── Dockerfile                # （Docker未使用時は無視可）
├── requirements.txt          # Python 依存ライブラリ一覧
├── README.md                 # 本ドキュメント
├── data_pipeline/            # データ取得・リサンプリング・特徴量生成
│   ├── resampler.py          # InformationDrivenResampler クラス
│   └── feature_store.py      # FracDiff やエントロピー計算
├── model/                    # モデル構築・学習・ラベリング
│   ├── regime_kalman.py      # 状態空間モデル（カルマンフィルタ）
│   ├── labeling.py           # トリプルバリア＋メタラベリング
│   └── trainer.py            # PurgedKFold CV & モデル保存/ロード
├── backtester/               # バックテストエンジンと評価指標
│   ├── engine.py             # シグナル→P&L計算
│   └── metrics.py            # シャープレシオ・最大ドローダウン
├── api/                      # 推論サーバ（FastAPI）
│   ├── schemas.py            # Pydanticスキーマ定義
│   └── main.py               # `/predict` エンドポイント
├── scheduler/                # バッチスケジューラー（Airflow/Prefect）
│   └── dags/                 # サンプルDAGスクリプト
├── monitoring/               # モニタリング構成（Prometheus/Grafana）
│   ├── grafana_dash.yml
│   └── prometheus_rules.yml
└── tests/                    # pytest 用ユニット・統合テスト
    ├── test_resampler.py
    ├── test_feature_store.py
    ├── test_regime_kalman.py
    ├── test_labeling.py
    ├── test_trainer.py
    ├── test_backtester.py
    └── test_api.py
````

---

## 🚀 ワークフロー概要

1. **データパイプライン** (`data_pipeline/`)

   * **リサンプリング**：ティックや分足を `InformationDrivenResampler` で情報駆動型バーに変換
   * **特徴量生成**：FracDiff、エントロピー、マイクロストラクチャ指標などを算出

2. **モデル構築** (`model/`)

   * **状態空間モデル** (`regime_kalman.py`)：2次元（トレンド×ボラティリティ）カルマンフィルタ
   * **ラベリング** (`labeling.py`)：トリプルバリア + メタラベリングによるサイド＆サイズ判定
   * **トレーナー** (`trainer.py`)：Purged K-Fold CV でモデル評価、学習済みモデルの保存／ロード

3. **バックテスト** (`backtester/`)

   * **エンジン** (`engine.py`)：シグナル→約定→P\&L累積計算
   * **指標** (`metrics.py`)：シャープレシオ、最大ドローダウン算出

4. **推論API** (`api/`)

   * `/predict` エンドポイントに価格時系列を送信し、レジーム推定結果（trend, vol）と売買シグナルを返却

5. **スケジューラー & モニタリング**

   * Airflow/Prefect で定期バッチ実行
   * Prometheus + Grafana でシステムとパフォーマンスを監視

---

## 📦 セットアップ手順

```bash
# 1. 仮想環境の作成
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

# 2. 依存ライブラリのインストール
pip install --upgrade pip
pip install -r requirements.txt

# 3. テスト実行
pytest -q
```

---

## 📚 参考文献

* Marcos López de Prado, *Advances in Financial Machine Learning*
* Pykalman, Statsmodels, mlfinlab ライブラリ

---

## 📝 ライセンス

MIT © Your Name

```
```
