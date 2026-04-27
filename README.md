# Titanic — Kaggle Competition

Задача бинарной классификации: предсказать выживаемость пассажиров Титаника.

## Структура проекта

```
.
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── base.py                    # Абстрактный базовый класс со стратифицированным CV
│   ├── logistic_regression.py
│   ├── svc.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── gradient_boosting.py
│   └── xgboost.py
├── neural_network/
│   ├── titanic_dataset.py         # PyTorch Dataset
│   ├── titanic_neural_network.py  # Полносвязная сеть: 32→16→1 с BN и Dropout
│   └── neural_network_learning.py # Цикл обучения
├── submission/
│   └── submission.csv
├── preprocessing.py               # Пайплайн признаков
├── config.py                      # Гиперпараметры моделей через OmegaConf
├── config_parser.py               # Конфиг → kwargs для моделей
├── main.py                        # Точка входа
├── eda.ipynb                      # Разведочный анализ данных
└── titanic.ipynb                  # Блокнот с экспериментами
```

## Пайплайн

```
train.csv / test.csv
        │
        ▼
  preprocessing.py
  ├── Извлечение звания (Initial) из имени
  ├── Заполнение пропусков Age и Fare медианами трейна
  ├── Cabin → буква палубы (Deck) → int
  ├── Embarked: пропуски → 'S'
  ├── FamilySize = SibSp + Parch + 1
  ├── FarePerPerson = Fare / FamilySize
  ├── AgeGroup: ребёнок / подросток / взрослый / пожилой
  ├── Флаг Alone
  ├── Взаимодействия WomanHighClass и ManThirdClass
  └── Удаление Name, Ticket, Cabin, PassengerId
        │
        ▼
   main.py обучает 6 sklearn-моделей + PyTorch-нейросеть
   Лучшая sklearn-модель (по CV-метрике) → submission.csv
```

## Модели

Все sklearn-модели используют единый стратифицированный 5-фолдовый CV из `BaseModel`. В финальный сабмит идёт модель с наилучшим CV-скором.

| Модель | Ключевые гиперпараметры | CV-метрика |
|---|---|---|
| Logistic Regression | C=0.01, solver=saga | roc_auc |
| SVC | C=1, kernel=rbf | accuracy |
| Decision Tree | max_depth=5, min_samples_leaf=10 | roc_auc |
| Random Forest | n_estimators=200, max_features=sqrt | roc_auc |
| Gradient Boosting | n_estimators=100, lr=0.1, subsample=0.8 | accuracy |
| XGBoost | n_estimators=200, max_depth=4, gamma=0.1 | roc_auc |

**Нейронная сеть** — полносвязная архитектура (вход → 32 → 16 → 1) с BatchNorm, Dropout (0.4 / 0.3) и Sigmoid на выходе. Обучается независимо и не участвует в отборе модели.

## Признаки

| Признак | Описание |
|---|---|
| `Pclass` | Класс билета (1 / 2 / 3) |
| `Sex` | male=0, female=1 |
| `Age` | Заполнен медианой трейна |
| `SibSp`, `Parch` | Супруги/братья, родители/дети |
| `Fare` | Стоимость билета, заполнена медианой трейна |
| `Embarked` | Порт: S=0, C=1, Q=2 |
| `Initial` | Звание из имени: Mr=0 Mrs=1 Miss=2 Master=3 Other=4 |
| `Deck` | Буква каюты: A–G → 1–7, Unknown → 8 |
| `FamilySize` | SibSp + Parch + 1 |
| `FarePerPerson` | Fare / FamilySize |
| `AgeGroup` | 1=ребёнок(0-12) 2=подросток(12-18) 3=взрослый(18-60) 4=пожилой(60+) |
| `Alone` | 1, если FamilySize == 1 |
| `WomanHighClass` | 1, если female и Pclass ≤ 2 |
| `ManThirdClass` | 1, если male и Pclass == 3 |

Подробный анализ, почему выбраны именно эти признаки — в `eda.ipynb`.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python main.py
```

Обучает все модели, выводит CV-скоры и сохраняет предсказания лучшей модели в `submission/submission.csv`.
