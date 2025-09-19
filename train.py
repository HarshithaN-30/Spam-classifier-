# train.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# --- Config ---
DATA_PATH = "data/spam.csv"
MODELS_DIR = "models"
EXPERIMENT_NAME = "Email_Spam_Classifier"
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Load dataset robustly (handles Kaggle variant) ---
df = pd.read_csv(DATA_PATH, encoding="latin-1", low_memory=False)

# Kaggle spam.csv sometimes contains extra unnamed columns — pick the two main cols:
if set(['v1','v2']).issubset(df.columns):
    df = df[['v1','v2']].rename(columns={'v1':'label','v2':'message'})
elif set(['label','message']).issubset(df.columns):
    df = df[['label','message']]
else:
    # fallback: assume first two columns are label and message
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']

# Map labels to 0/1
df['label'] = df['label'].map({'ham':0, 'spam':1}).astype(int)

# --- Basic preprocessing ---
df['message'] = df['message'].astype(str).str.strip()
df = df.dropna(subset=['message'])

# --- Train/test split ---
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Vectorize ---
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# --- Train model ---
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vect, y_train)
y_pred = model.predict(X_test_vect)

# --- Metrics ---
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

# --- Confusion matrix plot saved as artifact ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham','spam'])
disp.plot()
plt.title("Confusion Matrix")
os.makedirs("artifacts", exist_ok=True)
cm_path = os.path.join("artifacts", "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

# --- MLflow logging ---
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run():
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1", f1)

    # Log the trained sklearn model
    mlflow.sklearn.log_model(model, "spam_model")

    # Log the confusion matrix image as an artifact
    mlflow.log_artifact(cm_path, artifact_path="images")

# --- Save model & vectorizer locally for deployment ---
joblib.dump(model, os.path.join(MODELS_DIR, "spam_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))

print("Saved model and vectorizer to 'models/'")
print("MLflow run and artifacts saved to 'mlruns/' (local)")
