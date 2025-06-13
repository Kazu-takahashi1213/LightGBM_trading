import pandas as pd
import numpy as np
from backtester.engine import BacktestEngine
from backtester.metrics import sharpe_ratio, max_drawdown

def test_backtest_engine_and_metrics():
    idx = pd.date_range("2025-01-01", periods=10, freq="T")
    prices = pd.Series(np.linspace(100, 110, 10), index=idx)
    signals = pd.Series([1,0,1,0,1,0,1,0,1,0], index=idx)
    be = BacktestEngine(prices, signals, cost=0.0)
    equity = be.run()
    # 単調増加するなら最大ドローダウンは 0
    assert max_drawdown(equity) <= 0
    # リターン系列からシャープレシオを計算
    sr = sharpe_ratio(equity.diff().fillna(0))
    assert isinstance(sr, float)
