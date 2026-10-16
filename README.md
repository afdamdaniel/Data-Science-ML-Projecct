# Titanic Survival Prediction

A machine learning project that predicts which passengers survived the Titanic. It uses a scikit-learn pipeline for cleaning and encoding, then a tuned Random Forest classifier.

## Overview

This is the classic Kaggle "Titanic: Machine Learning from Disaster" problem. Given passenger details such as class, sex, age and fare, the model predicts a 0 or 1 for `Survived`.

The notebook covers:

1. Data exploration
2. A stratified train and test split
3. Custom preprocessing steps inside a scikit-learn `Pipeline`
4. Random Forest tuning with Grid Search
5. Evaluation on held-out data
6. Predictions for the Kaggle test file

## Dataset

- `train.csv`: 891 passengers, 12 columns, with the `Survived` label
- `test.csv`: 418 passengers, no label
- About 38% of passengers in the training file survived

| Column | Description |
|---|---|
| `PassengerId` | Row ID |
| `Survived` | Target (0 = no, 1 = yes) |
| `Pclass` | Ticket class (1, 2 or 3) |
| `Name` | Passenger name |
| `Sex` | male or female |
| `Age` | Age in years (714 of 891 values present) |
| `SibSp` | Number of siblings or spouses aboard |
| `Parch` | Number of parents or children aboard |
| `Ticket` | Ticket number |
| `Fare` | Ticket price |
| `Cabin` | Cabin number (mostly missing) |
| `Embarked` | Port of embarkation (C, Q or S) |

## Workflow

### 1. Exploration
- Looked at the first rows and summary statistics.
- Plotted a correlation heatmap of the numeric columns.

### 2. Stratified split
- Used `StratifiedShuffleSplit` with an 80/20 split: 712 training rows and 179 test rows.
- The split is stratified on `Survived`, `Pclass` and `Sex`, so both sets keep a similar mix of these groups.
- Compared the two sets with histograms.

### 3. Missing values
In the training split:
- `Age`: 137 missing
- `Cabin`: 552 missing
- `Embarked`: 1 missing
- No duplicate rows

### 4. Preprocessing pipeline
Three custom transformers, chained in a `Pipeline`:

| Step | What it does |
|---|---|
| `AgeImputer` | Fills missing `Age` with the mean |
| `FeatureEncoder` | One-hot encodes `Embarked` and `Sex` |
| `FeatureDropper` | Drops `Name`, `Ticket`, `Cabin` and the original text columns |

After the pipeline, the data is scaled with `StandardScaler`.

### 5. Model
- `RandomForestClassifier`
- `GridSearchCV` with 3-fold cross validation and accuracy as the score
- 36 combinations: `n_estimators` (10, 100, 200, 500), `max_depth` (None, 5, 10), `min_samples_split` (2, 3, 4)
- Best settings: `max_depth=5`, `min_samples_split=4`, `n_estimators=100`

### 6. Final predictions
- Retrained with the same grid search on all 891 labeled rows. Best settings: `max_depth=10`, `n_estimators=500`.
- Predicted survival for the 418 passengers in `test.csv`.
- Saved `prediction.csv` with columns `PassengerId` and `Survived`, the format Kaggle expects.

## Results

| Model | Accuracy | Evaluated on |
|---|---|---|
| Random Forest (tuned on the 80% split) | 0.810 | 179 held-out passengers |
| Random Forest (retrained on all data) | not measured | no held-out data left |

With only 179 test rows, one accuracy number can move by about 3 points from a different split. Treat 0.81 as a rough estimate.

## Limitations and next steps

- **`PassengerId` is used as a feature.** It is only a row number and should be dropped.
- **The `Embarked` columns are mislabeled.** The encoder sorts categories as C, Q, S, but the code names the columns C, S, Q. The column called `S` holds `Q` passengers and the reverse. The model still works, but any interpretation of these columns would be wrong. Use `encoder.get_feature_names_out()` to get the names.
- **Preprocessing is refit on the test data.** The scaler and the age mean are computed again on the test set. They should be fit on the training data only, then applied to the test data. Putting the scaler and model into the same `Pipeline` fixes this. (Random Forests do not need scaling, so this does not change the scores here.)
- **The missing `Fare` in `test.csv`** is filled from the row above it. Using the median fare of the passenger's class is better.
- **No `random_state`**, so results change on every run.
- **Feature ideas:** pull the title (Mr, Mrs, Miss) from `Name`, add a family size feature (`SibSp + Parch + 1`), use the cabin deck letter, and fill `Age` by title instead of one overall mean.
- Compare with Logistic Regression and Gradient Boosting, and use cross validation on all labeled data for the final score.

## Tech stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn


