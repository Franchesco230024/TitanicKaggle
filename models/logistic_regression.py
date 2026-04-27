from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic regression wrapped in a StandardScaler pipeline."""

    def __init__(self, C, solver, max_iter, l1_ratio, scoring, n_splits, shuffle, random_state):
        super().__init__(scoring, n_splits, shuffle, random_state)
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.l1_ratio = l1_ratio

    def _build_model(self):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state,
            ))
        ])
