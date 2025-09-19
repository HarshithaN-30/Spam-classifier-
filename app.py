# app.py
import os
import csv
from datetime import datetime
from flask import Flask, request, render_template, jsonify
import joblib

APP_PORT = int(os.environ.get("PORT", 5000))
MODEL_PATH = "models/spam_model.pkl"
VECT_PATH = "models/vectorizer.pkl"
LOGS_CSV = "logs/predictions.csv"

# Ensure logs folder exists
os.makedirs(os.path.dirname(LOGS_CSV) or ".", exist_ok=True)

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

app = Flask(__name__)

def append_log(message_text, prediction):
    header = ["timestamp", "message", "prediction"]
    exists = os.path.exists(LOGS_CSV)
    with open(LOGS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([datetime.utcnow().isoformat(), message_text, int(prediction)])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    if request.method == "POST":
        message = request.form.get("message", "")
        vect = vectorizer.transform([message])
        pred = model.predict(vect)[0]
        prediction_text = "🚨 SPAM" if pred == 1 else "✅ NOT SPAM"
        append_log(message, pred)
    return render_template("index.html", prediction=prediction_text)

@app.route("/predict", methods=["POST"])
def predict_api():
    """
    JSON API:
    POST /predict
    { "message": "your email text here" }
    Response:
    { "prediction": 1, "label": "spam" }
    """
    data = request.get_json(force=True)
    if not data or "message" not in data:
        return jsonify({"error": "send JSON with 'message' field"}), 400
    message = str(data["message"])
    vect = vectorizer.transform([message])
    pred = int(model.predict(vect)[0])
    append_log(message, pred)
    return jsonify({"prediction": pred, "label": "spam" if pred==1 else "not_spam"})

@app.route("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    # For local debugging
    app.run(host="0.0.0.0", port=APP_PORT, debug=False)
