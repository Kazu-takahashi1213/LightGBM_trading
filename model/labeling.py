import pandas as pd
from mlfinlab.labeling import get_events, add_vertical_barrier, apply_pt_sl
from mlfinlab.meta_labeling import ml_get_label

def generate_labels(
    prices: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: list = [1, 1],
    num_days: int = 5
) -> pd.DataFrame:
    """
    1. トリプルバリアでイベント取得
    2. 垂直バリア（時間制限）追加
    3. メタラベリングでサイド(+1/-1)とサイズを取得

    戻り値:
        labels: DataFrame(index=t_events, columns=['label'])
    """
    # トリプルバリア
    events = get_events(
        close=prices,
        t_events=t_events,
        pt_sl=pt_sl,
        target=prices.pct_change().abs(),
        min_ret=0.0,
        num_threads=4
    )
    # 垂直バリアを追加
    events = add_vertical_barrier(
        events=events,
        close=prices,
        num_days=num_days
    )
    # メタラベリング
    labels = ml_get_label(
        close=prices,
        events=events,
        pt_sl=pt_sl
    )
    return labels