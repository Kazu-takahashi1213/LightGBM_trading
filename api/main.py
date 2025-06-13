from fastapi import FastAPI
from typing import List
import pandas as pd
from model.regime_kalman import RegimeKalman
from model.trainer import ModelTrainer
from api.schemas import PricePoint, SignalResponse

app = FastAPI()

# モデルとトレーナーの初期化
trainer = ModelTrainer()
clf = trainer.load('models/model.pkl')

@app.post('/predict', response_model=SignalResponse)
def predict(data: List[PricePoint]):
    # リクエストを DataFrame に変換
    df = pd.DataFrame([p.dict() for p in data]).set_index('datetime')
    df.index = pd.to_datetime(df.index)

    # レジーム推定をフィッティング
    kf = RegimeKalman(df['price'])
    kf.fit_filter()
    states = kf.filter_states()

    # 1期間前の状態を特徴量にシグナル予測
    features = states.shift(1).dropna()
    signal = clf.predict(features)

    return SignalResponse(
        datetime=features.index.astype(str).tolist(),
        trend=features['trend'].tolist(),
        vol=features['vol'].tolist(),
        signal=signal.tolist()
    )