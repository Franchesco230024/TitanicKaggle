from sklearn.tree import DecisionTreeClassifier
from .base import BaseModel


class DecisionTreeModel(BaseModel):
    """Decision tree classifier."""

    def __init__(self, criterion, max_depth, min_samples_leaf, min_samples_split, scoring, n_splits, shuffle, random_state):
        super().__init__(scoring, n_splits, shuffle, random_state)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _build_model(self):
        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
