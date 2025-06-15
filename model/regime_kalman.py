import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel

class RegimeKalman(MLEModel):
    def __init__(self, endog: pd.Series, transition_cov: float = 0.01, obs_cov: float = 1.0):
        # Preserve index for output
        self._index = endog.index
        # 状態次元=2, k_posdef=2 を指定
        super().__init__(endog, k_states=2, k_posdef=2)
        self.transition_cov = transition_cov
        self.obs_cov = obs_cov
        # Initialize state with diffuse priors
        self.initialize_approximate_diffuse()

    @property
    def start_params(self):
        """Initial parameters for optimization."""
        return np.array([self.transition_cov, self.obs_cov])

    def transform_params(self, unconstrained):
        # 最適化パラメータを非負に制約
        return unconstrained ** 2

    def untransform_params(self, constrained):
        # 最適化後パラメータを元のスケールに戻す
        return np.sqrt(constrained)

    def update(self, params, **kwargs):
        # フィッティングごとに状態空間行列と共分散を再設定
        trans_cov, obs_cov = params

        # 観測行列
        self['design'] = np.array([[1.0, 0.0]])
        # 遷移行列（トレンドとボラティリティの関係）
        self['transition'] = np.array([[1.0, 1.0],
                                       [0.0, 1.0]])
        # 選択行列
        self['selection'] = np.eye(2)
        # 状態ノイズ共分散
        self['state_cov'] = np.eye(2) * trans_cov
        # 観測ノイズ共分散
        self['obs_cov']   = np.array([[obs_cov]])

    def fit_filter(self, **kwargs):
        # MLEでフィッティングし、スムーザー結果を保持
        res = super().fit(disp=False, **kwargs)
        self.smoother_results = res.smoother_results
        return res

    def filter_states(self) -> pd.DataFrame:
        # フィッティング後に smoothed_state を取り出し DataFrame で返す
        if not hasattr(self, 'smoother_results'):
            raise RuntimeError("まず fit_filter() を呼び出してください。")
        sm = self.smoother_results
        trend = sm.smoothed_state[0]
        vol   = sm.smoothed_state[1]
        return pd.DataFrame(
            {'trend': trend, 'vol': vol},
            index=self._index
        )