from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold, cross_val_score


class BaseModel(ABC):
    """Abstract base: owns CV loop, fit, and predict. Subclasses implement _build_model only."""

    def __init__(self, scoring, n_splits, shuffle, random_state):
        self.scoring = scoring
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.model = None
        self.score = None

    @abstractmethod
    def _build_model(self):
        """Return an unfitted sklearn-compatible estimator."""

    def fit(self, X, y):
        """Run stratified CV to populate self.score, then fit on full training data."""
        self.model = self._build_model()
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=self.scoring)
        self.score = scores.mean()
        print(f"{self.__class__.__name__} CV {self.scoring}: {self.score:.4f}")
        self.model.fit(X, y)

    def predict(self, X):
        """Return class predictions for X."""
        return self.model.predict(X)
