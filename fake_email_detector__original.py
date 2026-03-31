from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import pandas as pd
import math
import random
import logging
from collections import Counter
from flask import Flask, request, jsonify

# ✅ TF-IDF and Naïve Bayes model
vectorizer = TfidfVectorizer()
model = MultinomialNB()

def train_model(train_data, train_labels):
    X_train_tfidf = vectorizer.fit_transform(train_data)
    model.fit(X_train_tfidf, train_labels)

def predict_email(text):
    X_input = vectorizer.transform([text])
    return model.predict(X_input)[0]


# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Load Dataset from CSV File
def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV file must have 'text' and 'label' columns.")

    return df['text'].tolist(), df['label'].tolist()


# ✅ Convert Labels to Binary (1 = Fraud, 0 = Legit)
def preprocess_labels(labels):
    return [1 if str(label).lower() == 'fraud' else 0 for label in labels]


# ✅ Tokenization Function
def tokenize(text):
    if not isinstance(text, str):
        return []  # Return empty list if text is not a valid string

    return text.lower().split()


# ✅ Train-Test Split
def train_test_split_manual(data, labels, test_size=0.3):
    combined = list(zip(data, labels))
    random.shuffle(combined)
    split_index = int(len(combined) * (1 - test_size))
    train, test = combined[:split_index], combined[split_index:]
    train_data, train_labels = zip(*train)
    test_data, test_labels = zip(*test)
    return list(train_data), list(test_data), list(train_labels), list(test_labels)


# ✅ Train Naïve Bayes Model


# ✅ Prediction Function


# ✅ Evaluate Model


# ✅ Home Route (Fixes 404 Error)
@app.route('/')
def home():
    return "Fake Email Detector API is running!!🚀🏃‍➡️"


# ✅ API Endpoint for Predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        email_data = request.get_json()
        email_text = email_data.get('email', {}).get('body', '')

        if not email_text:
            return jsonify({"error": "No email body provided."}), 400

        predicted_label = predict_email(email_text)
        result = "Fraud" if predicted_label == 1 else "Legit"
        return jsonify({"prediction": result})

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ✅ Load Dataset & Train Model
file_path = os.path.join(os.path.dirname(__file__), 'email_dataset.csv')

try:
    data, labels = load_dataset(file_path)
    labels = preprocess_labels(labels)
    X_train, X_test, y_train, y_test = train_test_split_manual(data, labels)
    train_model(X_train, y_train)

    # Evaluate
    X_test_tfidf = vectorizer.transform(X_test)
    accuracy = model.score(X_test_tfidf, y_test)
    logging.info(f"✅ TF-IDF Model Accuracy: {accuracy * 100:.2f}%")

    logging.info(f"✅ Model Trained Successfully with Accuracy: {accuracy * 100:.2f}%")

except FileNotFoundError as e:
    logging.error(f"❌ {e}")
except ValueError as e:
    logging.error(f"❌ {e}")

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)

