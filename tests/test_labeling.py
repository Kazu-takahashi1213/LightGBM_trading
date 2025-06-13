import pandas as pd
import numpy as np
from model.labeling import generate_labels

def test_generate_labels_length():
    idx = pd.date_range("2025-01-01", periods=20, freq="T")
    prices = pd.Series(np.linspace(100, 120, 20), index=idx)
    # トリプルバリア用の t_events を適当に
    t_events = idx[::5]
    labels = generate_labels(prices, t_events, pt_sl=[0.01, 0.01], num_days=1)
    # ラベル数は t_events と同じ
    assert len(labels) == len(t_events)
    # 値は -1, 0, +1 のいずれか
    assert all(l in (-1, 0, 1) for l in labels.values)
