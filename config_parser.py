def get_model_config(config, model_type):
    """Return the model config block matching model_type."""
    return next(m for m in config.models if m.type == model_type)


def get_cv_params(config):
    """Return shared cross-validation parameters from global config."""
    return {
        "n_splits": config.cv.n_splits,
        "shuffle": config.cv.shuffle,
        "random_state": config.general.seed,
    }


def get_logistic_regression_params(config):
    """Return constructor kwargs for LogisticRegressionModel."""
    m = get_model_config(config, "logistic_regression")
    return {
        "C": m.C,
        "solver": m.solver,
        "max_iter": m.max_iter,
        "l1_ratio": m.l1_ratio,
        "scoring": m.scoring,
        **get_cv_params(config),
    }


def get_svc_params(config):
    """Return constructor kwargs for SVCModel."""
    m = get_model_config(config, "svc")
    return {
        "C": m.C,
        "kernel": m.kernel,
        "gamma": m.gamma,
        "scoring": m.scoring,
        **get_cv_params(config),
    }


def get_decision_tree_params(config):
    """Return constructor kwargs for DecisionTreeModel."""
    m = get_model_config(config, "decision_tree")
    return {
        "criterion": m.criterion,
        "max_depth": m.max_depth,
        "min_samples_leaf": m.min_samples_leaf,
        "min_samples_split": m.min_samples_split,
        "scoring": m.scoring,
        **get_cv_params(config),
    }


def get_random_forest_params(config):
    """Return constructor kwargs for RandomForestModel."""
    m = get_model_config(config, "random_forest")
    return {
        "n_estimators": m.n_estimators,
        "max_features": m.max_features,
        "max_depth": m.max_depth,
        "min_samples_split": m.min_samples_split,
        "min_samples_leaf": m.min_samples_leaf,
        "bootstrap": m.bootstrap,
        "scoring": m.scoring,
        **get_cv_params(config),
    }


def get_gradient_boosting_params(config):
    """Return constructor kwargs for GradientBoostingModel."""
    m = get_model_config(config, "gradient_boosting")
    return {
        "n_estimators": m.n_estimators,
        "learning_rate": m.learning_rate,
        "max_depth": m.max_depth,
        "min_samples_split": m.min_samples_split,
        "subsample": m.subsample,
        "scoring": m.scoring,
        **get_cv_params(config),
    }


def get_xgboost_params(config):
    """Return constructor kwargs for XGBoostModel."""
    m = get_model_config(config, "xgboost")
    return {
        "n_estimators": m.n_estimators,
        "max_depth": m.max_depth,
        "learning_rate": m.learning_rate,
        "subsample": m.subsample,
        "colsample_bytree": m.colsample_bytree,
        "gamma": m.gamma,
        "min_child_weight": m.min_child_weight,
        "scoring": m.scoring,
        **get_cv_params(config),
    }