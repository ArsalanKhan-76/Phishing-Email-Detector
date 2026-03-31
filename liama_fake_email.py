import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Step 1: Data Collection
# Creating a synthetic dataset
data = {
    'email': [
        'Congratulations, you have won a lottery! Click here to claim your prize.',
        'Hi John, can we schedule a meeting for tomorrow?',
        'Your account has been compromised. Please reset your password.',
        'Hello, we are excited to offer you a special discount on our products.',
        'Please find attached the report for the project.',
        'You have been selected for a free gift card. Click the link to claim.',
        'Reminder: Your appointment is scheduled for 10 AM tomorrow.',
        'Win a free vacation to the Bahamas! Click here to enter.',
        'Your invoice for the services is attached.',
        'Congratulations, you have been accepted into the program.',
        'Dear customer, your account has been suspended. Please contact us.',
        'Special offer: Get 50% off on your first purchase. Click here.',
        'Your Amazon order has been shipped. Track your package here.',
        'Urgent: Update your account information to avoid suspension.',
        'Join our exclusive club and get free rewards. Sign up now.',
        'Your bank statement is available. Download it here.',
        'Important: Verify your email address to continue using our service.',
        'Limited time offer: Get a free trial of our premium service.',
        'Your package has arrived. Collect it from the post office.',
        'You have been shortlisted for a job interview. Congratulations!'
    ],
    'label': [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]  # 1 for fraudulent, 0 for not fraudulent
}

df = pd.DataFrame(data)

# Step 2: Data Preprocessing
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['email'] = df['email'].apply(preprocess_text)

# Step 3: Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['email'])
y = df['label']

# Step 4: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred)}')