# Titanic Kaggle Competition

Решение соревнования [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).


## Feature Engineering

- `Initial` — титул из имени пассажира (Mr, Mrs, Miss, Master, Other)
- `Deck` — палуба из номера каюты
- `FamilySize` — размер семьи (SibSp + Parch + 1)
- `FarePerPerson` — цена билета делённая на размер семьи
- `AgeGroup` — возрастная группа (ребёнок / подросток / взрослый / пожилой)
- `Alone` — путешествует ли пассажир один
- `WomanHighClass` — женщина в 1-2 классе
- `ManThirdClass` — мужчина в 3 классе

Пропуски в `Age` и `Fare` заполняются медианой с тренировочной выборки. Медиана считается до сплита, чтобы не было утечки данных.

## Модели

Все классические модели обёрнуты в `sklearn.pipeline.Pipeline` со `StandardScaler` и подбором гиперпараметров через `GridSearchCV` с `StratifiedKFold(n_splits=5)`.

| Модель | Метрика |
|---|---|
| Logistic Regression | ROC-AUC |
| SVC | Accuracy |
| Decision Tree | ROC-AUC |
| Random Forest | ROC-AUC |
| Gradient Boosting | Accuracy |
| XGBoost | ROC-AUC |
| Voting Ensemble (RF + XGBoost + LR) | — |

## Нейросеть (PyTorch)

Многослойный перцептрон (MLP):

```
input(15) → Linear(32) → ReLU → BatchNorm1d → Dropout(0.4)
          → Linear(16) → ReLU → Dropout(0.3)
          → Linear(1)  → Sigmoid
```

- Оптимизатор: Adam (lr=1e-3)
- Loss: BCELoss
- Валидация через train_test_split (80/20)
- Устройство: MPS (Apple Silicon) / CPU

## Зависимости

```
pandas
numpy
scikit-learn
xgboost
torch
```

Установка:

```bash
pip install -r requirements.txt
```
