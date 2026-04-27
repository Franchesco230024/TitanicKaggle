from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .base import BaseModel


class SVCModel(BaseModel):
    """SVC wrapped in a StandardScaler pipeline."""

    def __init__(self, C, kernel, gamma, scoring, n_splits, shuffle, random_state):
        super().__init__(scoring, n_splits, shuffle, random_state)
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def _build_model(self):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                random_state=self.random_state,
            ))
        ])
