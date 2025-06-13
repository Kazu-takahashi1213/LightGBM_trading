import pandas as pd
import numpy as np
from data_pipeline.feature_store import FeatureStore

def test_fractional_diff_length_and_type():
    fs = FeatureStore(fracdiff_d=0.4)
    s = pd.Series(np.linspace(1, 5, 5))
    out = fs.fractional_diff(s)
    # 長さは変わらない
    assert isinstance(out, pd.Series)
    assert len(out) == len(s)

def test_entropy_placeholder():
    fs = FeatureStore()
    s = pd.Series(np.random.rand(100))
    # 実装前は NotImplementedError を投げるようにすると良い
    try:
        fs.entropy(s, window=10)
        got = True
    except NotImplementedError:
        got = False
    assert not got
