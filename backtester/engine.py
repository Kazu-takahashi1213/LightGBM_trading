import pandas as pd

class BacktestEngine:
    def __init__(self, prices: pd.Series, signals: pd.Series, cost: float = 0.0005):
        """
        prices: 終値時系列
        signals: {－1,0,+1} の売買シグナル
        cost: 取引コスト（割合）
        """
        self.prices = prices
        self.signals = signals
        self.cost = cost

    def run(self) -> pd.Series:
        """
        シグナルに基づき P&L（累積リターン）を計算して返す
        """
        # 価格リターン × シグナル
        ret = self.prices.pct_change().shift(-1) * self.signals
        # 取引発生時のコスト差し引き
        trades = self.signals.diff().abs()
        ret -= trades * self.cost
        # 累積リターン
        return ret.cumsum()