import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mlfinlab.cross_validation import PurgedKFold
from sklearn.metrics import accuracy_score
import joblib

class ModelTrainer:
    def __init__(self, model: RandomForestClassifier = None):
        # ランダムフォレストをデフォルトモデルとする
        self.model = model or RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=42
        )

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        embargo_td: pd.Timedelta = pd.Timedelta('1 days')
    ) -> float:
        """
        Purged K-Fold CV で汎化精度を測定し、平均正解率を返す
        """
        pkf = PurgedKFold(n_splits=n_splits, embargo_td=embargo_td)
        scores = []
        for train_idx, test_idx in pkf.split(X, y):
            clf = self.model
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = clf.predict(X.iloc[test_idx])
            scores.append(accuracy_score(y.iloc[test_idx], preds))
        return float(np.mean(scores))

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        全データでモデルを学習
        """
        self.model.fit(X, y)
        return self.model

    def save(self, path: str):
        """
        学習済みモデルをファイル保存
        """
        joblib.dump(self.model, path)

    def load(self, path: str):
        """
        モデルをロードして self.model にセット
        """
        self.model = joblib.load(path)
        return self.model