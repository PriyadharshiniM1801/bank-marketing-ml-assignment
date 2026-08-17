# Bank Marketing Classification — ML Assignment 2

## a. Problem Statement

The objective is to build classification models that predict whether a bank customer will subscribe to a term deposit. The project implements multiple supervised classification algorithms, evaluates them using six required metrics, and provides an interactive Streamlit application for testing the trained models.

## b. Dataset Description

Dataset: **Bank Marketing — bank-additional-full.csv**

- Records: **41,188**
- Input features: **20**
- Target: **y**
- Target classes: `yes` and `no`
- The dataset contains demographic, campaign, contact and economic indicators.

The target is encoded as `1` for `yes` and `0` for `no` during model training.

## c. GitHub Repository Link

**Replace this placeholder with your GitHub repository URL after creating the repository:**

`https://github.com/YOUR_USERNAME/bank-marketing-ml-assignment`

## d. Models Used

Five models are implemented as specified in the assignment:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9166 | 0.9424 | 0.7118 | 0.4364 | 0.5411 | 0.5162 |\n| Decision Tree | 0.9165 | 0.9315 | 0.6613 | 0.5302 | 0.5885 | 0.5468 |\n| kNN | 0.9082 | 0.8993 | 0.6419 | 0.4192 | 0.5072 | 0.4717 |\n| Naive Bayes | 0.8203 | 0.8393 | 0.3495 | 0.6907 | 0.4642 | 0.4009 |\n| Random Forest | 0.8926 | 0.9509 | 0.5135 | 0.8793 | 0.6484 | 0.6199 |\n
### Observations on Model Performance

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Provides a strong baseline and generally offers a good balance between precision and recall. |
| Decision Tree | Captures nonlinear relationships but can be sensitive to tree depth and may overfit without constraints. |
| kNN | Uses similarity between observations and is affected by feature scaling and the high-dimensional encoded feature space. |
| Naive Bayes | Fast and simple, but its conditional-independence assumption limits performance when predictors are correlated. |
| Random Forest | Combines many decision trees and generally provides robust nonlinear classification performance. |
| **Overall Winner** | **Random Forest**, based primarily on the highest F1 score in this experiment. |

### Data Preprocessing

- Numerical missing values are imputed with the median.
- Categorical missing values are imputed with the most frequent value.
- Categorical variables are one-hot encoded.
- Numerical variables are standardized.
- The dataset is split into 80% training and 20% testing using stratification and `random_state=42`.

### Streamlit Features

The application provides:

- CSV test-data upload
- ML model selection dropdown
- Predictions and subscription probabilities
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Downloadable prediction CSV

### Repository Structure

```text
bank-marketing-ml-assignment/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── bank-additional-full.csv
├── test_data.csv
├── test_data_demo.csv
├── model_metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

### Running Locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

### Deployment

Push the project to GitHub, then create a Streamlit Community Cloud application using `app.py` as the main file.

**Important:** Replace the GitHub placeholder above with your actual repository link before submitting the PDF.
