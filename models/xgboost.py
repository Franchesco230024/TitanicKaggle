from xgboost import XGBClassifier
from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost classifier."""

    def __init__(self, n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, min_child_weight, scoring, n_splits, shuffle, random_state):
        super().__init__(scoring, n_splits, shuffle, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight

    def _build_model(self):
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )
