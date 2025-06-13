import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series, freq: int = 252*78*60) -> float:
    """
    年率化シャープレシオ
    returns: リターン時系列（日次換算ではなく分足換算）
    freq: 年間取引頻度（例: 252営業日×78本/日×60分）
    """
    return np.sqrt(freq) * returns.mean() / returns.std()

def max_drawdown(equity: pd.Series) -> float:
    """
    最大ドローダウン
    equity: 累積リターン（資産曲線）
    """
    cum_max = equity.cummax()
    drawdown = (equity - cum_max) / cum_max
    return drawdown.min()