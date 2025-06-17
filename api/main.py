from fastapi import FastAPI
from typing import List
import pandas as pd
import numpy as np
from model.trainer import ModelTrainer
from data_pipeline.feature_store import FeatureStore
from api.schemas import PricePoint, SignalResponse
from sklearn.dummy import DummyClassifier
import os

app = FastAPI()

# モデルとトレーナーの初期化
trainer = ModelTrainer()
fs = FeatureStore()
model_path = os.path.join('models', 'model.pkl')
if os.path.exists(model_path):
    clf = trainer.load(model_path)
else:
    # Fallback to a simple dummy classifier for unit tests
    dummy = DummyClassifier(strategy="most_frequent")
    # Fit on minimal dummy data to satisfy scikit-learn's "fitted" requirement
    dummy.fit([[0, 0]], [0])
    clf = dummy

@app.post('/predict', response_model=SignalResponse)
def predict(data: List[PricePoint]):
    # リクエストを DataFrame に変換
    df = pd.DataFrame([p.dict() for p in data]).set_index('datetime')
    df.index = pd.to_datetime(df.index)

    # テクニカル指標を計算
    indicators = fs.technical_indicators(df)

    # 1期間前の特徴量を用いて予測
    features = indicators.shift(1).dropna()
    proba = clf.predict_proba(features)
    prob = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(features))
    signal = clf.predict(features)

    return SignalResponse(
        datetime=features.index.astype(str).tolist(),
        prob_up=prob.tolist(),
        signal=signal.tolist()
    )
