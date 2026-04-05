# Phishing Email Detection API 🛡️

A REST API that automatically detects phishing and fraudulent emails 
using Natural Language Processing (NLP) and Machine Learning.

## How It Works
1. Email text is sent to the API via a POST request in JSON format
2. TF-IDF vectorizer extracts meaningful text features
3. Naive Bayes classifier predicts whether the email is Phishing or Legitimate
4. Result is returned instantly as a JSON response

## Tech Stack
- Python, Flask
- Scikit-learn (TF-IDF Vectorization + Multinomial Naive Bayes)
- Pandas (Dataset handling)
- Postman (API Testing)

## Project Structure
phishing-email-detector/
├── fake_email_detector__original.py   ← Flask API + ML Model
├── tfidf_model.py                     ← TF-IDF Model logic
├── email_dataset.csv                  ← Training dataset
├── requirements.txt                   ← Dependencies
└── README.md                          ← You are here

## API Usage

### Endpoint
POST /predict

### Request Format (JSON)
{
  "email": {
    "body": "Congratulations! You have won a lottery prize. Click here to claim."
  }
}

### Response Format
{
  "prediction": "Fraud"
}

## Security Relevance
Phishing emails are responsible for over 90% of cyberattacks worldwide.
This project demonstrates how ML-based threat detection can be applied in:
- Email security gateways
- SOC (Security Operations Center) automation
- Spam filtering systems

## Model Performance
- Algorithm: Multinomial Naive Bayes
- Feature Extraction: TF-IDF Vectorization
- Train/Test Split: 70/30
- Accuracy: check terminal logs when running

## How to Run Locally
1. Clone this repo
2. Install dependencies: pip install -r requirements.txt
3. Run the app: python fake_email_detector__original.py
4. Open Postman and send a POST request to http://127.0.0.1:5000/predict

## ⚠️ Known Limitations
- Model struggles with social engineering emails that use 
  professional language (e.g. fake job offers)
- URL analysis is not yet implemented
- Training dataset lacks job scam examples

## Author
Arsalan Khan Pathan | B-Tech Student | Cybersecurity Enthusiast
