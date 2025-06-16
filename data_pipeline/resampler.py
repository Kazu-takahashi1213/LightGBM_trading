import pandas as pd
import numpy as np

class InformationDrivenResampler:
    """
    一定の volume_threshold を超えたらバーを確定させる情報駆動型バー生成器。
    """

    def __init__(self, volume_threshold: float):
        self.volume_threshold = volume_threshold

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            index が日時、カラムに 'price' と 'volume' を含む DataFrame
        
        Returns
        -------
        pd.DataFrame
            columns=['open','high','low','close','volume'] のバー DataFrame
        """
        bars = []
        cum_vol = 0.0
        start_idx = df.index[0]

        for timestamp, row in df.iterrows():
            cum_vol += row['volume']
            if cum_vol >= self.volume_threshold:
                slice_df = df.loc[start_idx:timestamp]
                open_  = slice_df['price'].iloc[0]
                high_  = slice_df['price'].max()
                low_   = slice_df['price'].min()
                close_ = slice_df['price'].iloc[-1]
                vol_   = slice_df['volume'].sum()
                bars.append({
                    'datetime': timestamp,
                    'open': open_, 'high': high_, 'low': low_, 'close': close_, 'volume': vol_
                })
                # For this simplified example we only generate a single bar.
                break

        bars_df = pd.DataFrame(bars).set_index('datetime')
        return bars_df