from sklearn.ensemble import GradientBoostingClassifier
from .base import BaseModel


class GradientBoostingModel(BaseModel):
    """Sklearn GradientBoostingClassifier."""

    def __init__(self, learning_rate, max_depth, min_samples_split, n_estimators, subsample, scoring, n_splits, shuffle, random_state):
        super().__init__(scoring, n_splits, shuffle, random_state)
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.subsample = subsample

    def _build_model(self):
        return GradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
        )
