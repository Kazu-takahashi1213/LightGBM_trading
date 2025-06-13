import pandas as pd
from data_pipeline.resampler import InformationDrivenResampler

def test_resampler_simple():
    idx = pd.date_range("2025-01-01", periods=5, freq="T")
    df = pd.DataFrame({
        "price": [1, 2, 3, 4, 5],
        "volume": [5, 5, 5, 5, 5]
    }, index=idx)

    resampler = InformationDrivenResampler(volume_threshold=10)
    bars = resampler.transform(df)
    # 最初の2行で10を超え、1バー生成。残りは不足なので1本だけ
    assert len(bars) == 1
    assert bars.iloc[0]['open'] == 1
    assert bars.iloc[0]['close'] == 2