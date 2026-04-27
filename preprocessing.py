import re
import pandas as pd
import numpy as np


def get_train_fare_median(df: pd.DataFrame):
    """Return median fare from the given dataframe."""
    return df['Fare'].median()

def get_train_age_median(df: pd.DataFrame):
    """Return median age from the given dataframe."""
    return df['Age'].median()

def add_initial_feature(df: pd.DataFrame):
    """Extract title from Name, normalise rare titles, encode as integer."""
    df['Initial'] = df['Name'].str.extract(r'([A-Za-z]+)\.')

    df['Initial'] = df['Initial'].replace(
        ['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mrs']
    )

    df['Initial'] = df['Initial'].replace({
        'Mr': 0,
        'Mrs': 1,
        'Miss': 2,
        'Master': 3,
        'Other': 4
    }).astype(int)

    return df

def feature_engineering(df: pd.DataFrame, fare_median, age_median):
    """Impute missing values and build derived features.

    Uses train-set medians for Age and Fare to avoid data leakage on the test set.
    """
    df['Age'] = df['Age'].fillna(age_median)

    df['Cabin'] = df['Cabin'].fillna("U0")

    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

    df['Deck'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    df['Deck'] = df['Deck'].map(deck)

    df['Deck'] = df['Deck'].fillna(0)

    df['Deck'] = df['Deck'].astype(int)

    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'] = (df['Fare'].fillna(fare_median))

    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, np.inf], labels=False) + 1

    df['Alone'] = 0
    df.loc[df['FamilySize'] == 1, 'Alone'] = 1

    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1}).astype(int)
    df['Embarked'] = df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    df['WomanHighClass'] = ((df['Sex'] == 1) & (df['Pclass'] <= 2)).astype(int)
    df['ManThirdClass'] = ((df['Sex'] == 0) & (df['Pclass'] == 3)).astype(int)

    df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    return df

def get_target(train: pd.DataFrame):
    """Return the Survived column as the target series."""
    return train["Survived"]

def preprocessing(train: pd.DataFrame, test: pd.DataFrame):
    """Run full preprocessing pipeline on train and test sets.

    Computes statistics on train only, then applies them to both splits.
    Returns (X_train, X_test) as feature-only dataframes.
    """
    X_train = add_initial_feature(train)
    X_train.pop("Survived")
    X_test = add_initial_feature(test)
    fare_median = get_train_fare_median(train)
    age_median = get_train_age_median(train)

    X_train = feature_engineering(X_train, fare_median, age_median)
    X_test = feature_engineering(X_test, fare_median, age_median)

    return X_train, X_test
