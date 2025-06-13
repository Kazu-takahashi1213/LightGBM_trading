import pandas as pd
from fracdiff import Fracdiff

class FeatureStore:
    def __init__(self, fracdiff_d: float = 0.4):
        self.fracdiff = Fracdiff(d=fracdiff_d)

    def fractional_diff(self, series: pd.Series) -> pd.Series:
        """長期記憶を保ったまま定常化するフラクショナル差分"""
        return pd.Series(self.fracdiff.fit_transform(series.values), index=series.index)

    def entropy(self, series: pd.Series, window: int = 20) -> pd.Series:
        """移動ウィンドウ上のエントロピー計算"""
        # TODO: 実装 or scipy.stats.entropy を利用
        pass

    def microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """スプレッドやトレード不均衡などマイクロストラクチャ指標"""
        # TODO: 実装
        pass
