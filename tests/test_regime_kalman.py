import pandas as pd
import numpy as np
from model.regime_kalman import RegimeKalman

def test_simple_kalman_runs():
    # 緩やかなランダムウォークデータ
    idx = pd.date_range("2025-01-01", periods=50, freq="T")
    prices = pd.Series(np.cumsum(np.random.randn(50)), index=idx)
    kf = RegimeKalman(prices)
    res = kf.fit_filter()
    states = kf.filter_states()
    # trend, vol のカラムがある
    assert 'trend' in states.columns and 'vol' in states.columns
    # インデックスが一致
    assert list(states.index) == list(prices.index)
