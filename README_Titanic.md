# Titanic Survival Prediction using Machine Learning

A binary classification project using the **Titanic dataset** to predict passenger survival. The notebook primarily demonstrates **Logistic Regression** and compares it with KNN, Gaussian Naive Bayes, and Decision Tree classifiers.

## Objective

Predict whether a Titanic passenger survived (`1`) or did not survive (`0`) using demographic and travel-related features.

## Dataset

The notebook uses the Titanic dataset provided by Seaborn:

```python
sns.load_dataset("titanic")
```

Relevant features are selected after removing redundant or leakage-prone columns.

## Data Preprocessing

The project performs:

1. Removal of columns such as `deck`, `embark_town`, `alive`, `class`, `who`, and `adult_male`.
2. Missing-value handling for `age`.
3. Removal of rows with missing `embarked` values.
4. Label encoding of `sex` and `embarked`.
5. Separation of features (`X`) and target (`y`).
6. Stratified 80/20 train-test split.
7. Standardization of numerical input features.

## Target Variable

`survived`

- `0` → Did not survive
- `1` → Survived

## Models Implemented

### 1. Logistic Regression

Used as the primary baseline classification model.

### 2. K-Nearest Neighbors (KNN)

Implemented with `n_neighbors=5`.

### 3. Gaussian Naive Bayes

Implemented using `GaussianNB`.

### 4. Decision Tree

Implemented with:

- Criterion: Gini
- Maximum depth: 3
- Random state: 42

## Evaluation

The models are evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score
- Classification Report

## Recorded Results

The original notebook recorded the following test accuracies:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 80.34% |
| KNN | 79.21% |
| Gaussian Naive Bayes | 77.53% |
| Decision Tree | 82.02% |

These figures are retained as the original notebook's recorded outputs. Because the cleaned notebook uses more consistent preprocessing, rerunning it may produce slightly different values.

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

## Repository Structure

```text
Titanic-Survival-Prediction/
│
├── Titanic_Survival_Prediction_Logistic_Regression.ipynb
├── README.md
└── requirements.txt
```

## How to Run

### Google Colab

Open the notebook in Google Colab and run the cells from top to bottom. The Titanic dataset is loaded directly through Seaborn, so no separate CSV upload is required.

### Local Environment

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook using Jupyter Notebook or JupyterLab.

## Notes

Although Logistic Regression is the main model highlighted in this project, the notebook also contains three additional classical classifiers to provide a simple performance comparison.
