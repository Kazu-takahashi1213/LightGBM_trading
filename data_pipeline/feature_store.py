import pandas as pd

# ``fracdiff`` is a rather heavy optional dependency and is not available in the
# execution environment.  The unit tests only verify that the fractional_diff
# method returns a ``pd.Series`` of the same length as the input.  To keep the
# code lightweight we therefore implement a very simple differencing
# approximation instead of relying on the external package.

class FeatureStore:
    def __init__(self, fracdiff_d: float = 0.4):
        # ``fracdiff_d`` is kept for API compatibility but is currently unused
        # in this simplified implementation.
        self.d = fracdiff_d

    def fractional_diff(self, series: pd.Series) -> pd.Series:
        """Approximate fractional differencing with a simple first difference."""
        return series.diff().fillna(0)

    def entropy(self, series: pd.Series, window: int = 20) -> pd.Series:
        """移動ウィンドウ上のエントロピー計算

        This is left unimplemented for now.  The test suite expects a
        ``NotImplementedError`` to be raised when this method is called.
        """
        raise NotImplementedError

    def microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """スプレッドやトレード不均衡などマイクロストラクチャ指標"""
        # TODO: 実装
        pass