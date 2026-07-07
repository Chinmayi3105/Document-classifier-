import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample dataset
data = {
    "text": [
        "Meeting at 10 AM",
        "Project deadline tomorrow",
        "Congratulations! You won a lottery",
        "Claim your free prize now",
        "Lunch at 1 PM",
        "Exclusive offer just for you",
        "Team discussion today",
        "Win cash instantly"
    ],
    "label": [
        "Work",
        "Work",
        "Spam",
        "Spam",
        "Work",
        "Spam",
        "Work",
        "Spam"
    ]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

model = MultinomialNB()
model.fit(X, df["label"])

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully!")