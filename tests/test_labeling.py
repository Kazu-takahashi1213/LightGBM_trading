import pandas as pd
import numpy as np

def simple_labeling(close, events, pt_sl):
    """
    簡易版トリプルバリアによるラベリング
    """
    out = pd.Series(index=events.index, dtype='int')
    for t in events.index:
        t1 = events.loc[t, 't1']
        if pd.isna(t1) or t1 not in close:
            out[t] = 0
            continue
        price_slice = close[t:t1]
        ret = (price_slice / close[t]) - 1
        if ret.max() > pt_sl[0]:
            out[t] = 1
        elif ret.min() < -pt_sl[1]:
            out[t] = -1
        else:
            out[t] = 0
    return pd.DataFrame({'label': out})

def generate_labels(prices, t_events, pt_sl=[0.01, 0.01], num_days=1):
    """
    簡易トリプルバリアラベリング
    """
    t1 = prices.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < len(prices)]
    t1_index = prices.index[t1]
    events = pd.DataFrame(index=t_events)
    events['t1'] = t1_index[:len(events)]
    return simple_labeling(prices, events, pt_sl)

def test_generate_labels_length():
    idx = pd.date_range("2025-01-01", periods=20, freq="T")
    prices = pd.Series(np.linspace(100, 120, 20), index=idx)
    t_events = idx[::5]
    labels = generate_labels(prices, t_events, pt_sl=[0.01, 0.01], num_days=1)
    assert len(labels) == len(t_events)
    assert all(l in (-1, 0, 1) for l in labels['label'].values)

