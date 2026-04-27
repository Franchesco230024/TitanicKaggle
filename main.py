import pandas as pd

from config import config
from preprocessing import preprocessing, get_target

from config_parser import (
    get_logistic_regression_params,
    get_svc_params,
    get_decision_tree_params,
    get_random_forest_params,
    get_gradient_boosting_params,
    get_xgboost_params,
)
from models import (
    LogisticRegressionModel,
    SVCModel,
    DecisionTreeModel,
    RandomForestModel,
    GradientBoostingModel,
    XGBoostModel,
)
from neural_network import NeuralNetworkLearning

MODEL_REGISTRY = {
    "logistic_regression": (LogisticRegressionModel, get_logistic_regression_params),
    "svc": (SVCModel, get_svc_params),
    "decision_tree": (DecisionTreeModel, get_decision_tree_params),
    "random_forest": (RandomForestModel, get_random_forest_params),
    "gradient_boosting": (GradientBoostingModel, get_gradient_boosting_params),
    "xgboost": (XGBoostModel, get_xgboost_params),
}


def train_neural_network(X_train, y_train, X_test):
    """Train the PyTorch neural network and return predictions on X_test."""
    neural_network = NeuralNetworkLearning()
    model = neural_network.fit(X_train, y_train)
    predictions = neural_network.predict(model, X_test)
    return predictions

def train_all_models(X_train, y_train):
    """Train every model listed in config and return a dict of fitted model objects."""
    trained_models = {}

    for model_cfg in config.models:
        model_type = model_cfg.type

        if model_type not in MODEL_REGISTRY:
            print(f"Unknown model type: {model_type}, skipping")
            continue

        model_class, get_params = MODEL_REGISTRY[model_type]
        params = get_params(config)

        print(f"\n>>> Training {model_type}")
        model = model_class(**params)
        model.fit(X_train, y_train)

        trained_models[model_type] = model

    return trained_models

def get_best_model(trained_models):
    """Return the model with the highest CV score."""
    best_name = max(trained_models, key=lambda name: trained_models[name].score)
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} with CV score: {best_model.score:.4f}")
    return best_model

def predict(model, X_test):
    """Return predictions from the given model."""
    predictions = model.predict(X_test)
    return predictions

def load_data():
    """Load train and test CSVs from paths defined in config."""
    train = pd.read_csv(config.paths.path_to_train)
    test = pd.read_csv(config.paths.path_to_test)
    return train, test

def get_submission(predictions, test):
    """Save predictions as a Kaggle submission CSV."""
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv(config.paths.path_to_submission, index=False)
    print(f"Submission saved")

def main():
    """Run the full pipeline: load → preprocess → train → select best → submit."""
    train, test = load_data()

    y_train = get_target(train)
    X_train, X_test = preprocessing(train, test)

    trained_models = train_all_models(X_train, y_train)
    train_neural_network(X_train, y_train, X_test)

    best_model = get_best_model(trained_models)
    predictions = predict(best_model, X_test)
    get_submission(predictions, test)

if __name__ == "__main__":
    main()
