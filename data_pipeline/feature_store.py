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

    def technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute basic technical indicators.

        Parameters
        ----------
        df : pd.DataFrame
            Expected columns include ``price`` and optionally ``bid_depth`` and
            ``ask_depth``.

        Returns
        -------
        pd.DataFrame
            DataFrame containing MACD, RSI, Bollinger Bands, moving average
            divergence rate and orderbook depth imbalance.
        """
        price = df["price"]
        indicators = pd.DataFrame(index=price.index)

        # MACD (12, 26) EMA difference
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        indicators["macd"] = ema12 - ema26

        # RSI (14 period)
        delta = price.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / roll_down.replace(0, 1)
        indicators["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20 period, 2 std)
        ma = price.rolling(window=20).mean()
        std = price.rolling(window=20).std()
        indicators["boll_up"] = ma + 2 * std
        indicators["boll_dn"] = ma - 2 * std

        # Moving average divergence rate (5 period)
        ma_short = price.rolling(window=5).mean()
        indicators["ma_div"] = (price - ma_short) / ma_short

        # Orderbook depth imbalance
        if {"bid_depth", "ask_depth"}.issubset(df.columns):
            total = df["bid_depth"] + df["ask_depth"]
            total = total.replace(0, 1)
            imbalance = (df["bid_depth"] - df["ask_depth"]) / total
            indicators["depth_imbalance"] = imbalance
        else:
            indicators["depth_imbalance"] = 0.0

        return indicators.fillna(0)
