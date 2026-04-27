from omegaconf import OmegaConf

config = {
    "general": {
        "seed": 42
    },
    "cv": {
        "n_splits": 5,
        "shuffle": True
    },
    "models": [
        {
            "type": "logistic_regression",
            "max_iter": 1000,
            "solver": "saga",
            "C": 0.01,
            "l1_ratio": 0,
            "scoring": "roc_auc",
        },
        {
            "type": "svc",
            "C": 1,
            "gamma": "scale",
            "kernel": "rbf",
            "scoring": "accuracy",
        },
        {
            "type": "decision_tree",
            'criterion': 'gini',
            'max_depth': 5,
            'min_samples_leaf': 10,
            'min_samples_split': 2,
            "scoring": "roc_auc"
        },
        {
            "type": "random_forest",
            'bootstrap': True,
            'max_depth': None,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'min_samples_split': 5,
            'n_estimators': 200,
            "scoring": "roc_auc"
        },
        {
            "type": "gradient_boosting",
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'n_estimators': 100,
            'subsample': 0.8,
            "scoring": "accuracy"
        },
        {
            "type": "xgboost",
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'learning_rate': 0.1,
            'max_depth': 4,
            'min_child_weight': 5,
            'n_estimators': 200,
            'subsample': 0.8,
            "scoring": "roc_auc"
        }
    ],
    "paths": {
        "path_to_train": "./data/train.csv",
        "path_to_test": "./data/test.csv",
        "path_to_submission": "./submission/submission.csv"
    }
}

config = OmegaConf.create(config)