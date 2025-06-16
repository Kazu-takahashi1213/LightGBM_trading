import pandas as pd


def generate_labels(
    prices: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: list | None = None,
    num_days: int = 5
) -> pd.Series:
    """Generate simple directional labels.

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by datetime.
    t_events : pd.DatetimeIndex
        Event timestamps to label.
    pt_sl : list | None, optional
        Unused in this lightweight implementation. Present for API compatibility.
    num_days : int, optional
        Unused in this lightweight implementation.

    Returns
    -------
    pd.Series
        Series of labels (+1, 0 or -1) indexed by ``t_events``.
    """
    # Compute forward return for each timestamp in ``t_events``
    future_ret = prices.pct_change().shift(-1)
    labels = future_ret.loc[t_events].fillna(0)
    return labels.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

