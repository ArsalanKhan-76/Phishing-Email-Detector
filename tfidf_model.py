from sklearn.feature_extraction.text import TfidfVectorizer

text = ["This is a sample email", "Another example for fraud detection"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text)

print(X.toarray())
