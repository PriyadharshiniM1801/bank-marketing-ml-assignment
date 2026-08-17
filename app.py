import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Marketing Classifier", page_icon="📊", layout="wide")

MODEL_DIR = Path("model")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.title("📊 Bank Marketing Classification")
st.write(
    "Interactive classification of whether a bank customer will subscribe "
    "to a term deposit (target: **y**)."
)

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Select ML Model", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader("Upload test data CSV", type=["csv"])

if uploaded is not None:
    data = pd.read_csv(uploaded)
else:
    demo_path = Path("test_data_demo.csv")
    data = pd.read_csv(demo_path) if demo_path.exists() else None
    if data is not None:
        st.info("No CSV uploaded. Showing the included demo test data.")

if data is None:
    st.warning("Upload a CSV file to continue.")
    st.stop()

st.subheader("Input Data")
st.dataframe(data.head(20), use_container_width=True)

target = "y"
if target not in data.columns:
    st.warning("The uploaded CSV does not contain the target column 'y'. Predictions will be shown without evaluation metrics.")
    X_input = data.copy()
    y_true = None
else:
    X_input = data.drop(columns=[target])
    y_true = (data[target].astype(str).str.lower() == "yes").astype(int)

model = load_model(MODEL_DIR / MODEL_FILES[model_name])
pred = model.predict(X_input)
proba = model.predict_proba(X_input)[:, 1]

pred_labels = np.where(pred == 1, "yes", "no")
result = data.copy()
result["Predicted_y"] = pred_labels
result["Subscription_Probability"] = np.round(proba, 4)

st.subheader(f"Predictions — {model_name}")
st.dataframe(result.head(50), use_container_width=True)

if y_true is not None:
    st.subheader("Evaluation Metrics")
    metrics = {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, proba) if y_true.nunique() == 2 else np.nan,
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1 Score": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }

    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["No", "Yes"])
    ax.set_yticks([0, 1], ["No", "Yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y_true, pred, target_names=["No", "Yes"], output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.download_button(
    "Download Predictions CSV",
    result.to_csv(index=False).encode("utf-8"),
    "bank_marketing_predictions.csv",
    "text/csv"
)
