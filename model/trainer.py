import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import joblib
from typing import Optional

class ModelTrainer:
    def __init__(self, model: LGBMClassifier | None = None):
        """Initialize trainer with a LightGBM classifier by default."""
        self.model = model or LGBMClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        embargo_td: Optional[pd.Timedelta] = None,
    ) -> float:
        """Perform simple K-Fold cross validation and return the mean accuracy."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in kf.split(X):
            clf = self.model
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = clf.predict(X.iloc[test_idx])
            scores.append(accuracy_score(y.iloc[test_idx], preds))
        return float(np.mean(scores))

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model on all data."""
        self.model.fit(X, y)
        return self.model

    def save(self, path: str):
        """Persist the trained model to file."""
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load a saved model and assign it to ``self.model``."""
        self.model = joblib.load(path)
        return self.model
