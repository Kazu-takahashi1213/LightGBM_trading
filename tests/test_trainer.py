import pandas as pd
import numpy as np
from model.trainer import ModelTrainer

def test_trainer_cv_and_train():
    # ダミー特徴量とラベル
    X = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100)
    })
    y = pd.Series(np.random.choice([0,1], size=100))
    mt = ModelTrainer()
    score = mt.cross_validate(X, y, n_splits=3, embargo_td=pd.Timedelta('1T'))
    assert 0.0 <= score <= 1.0
    model = mt.train(X, y)
    # predict ができる
    preds = model.predict(X)
    assert len(preds) == len(y)
