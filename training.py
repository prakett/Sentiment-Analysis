from numpy import test
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_csv('dataset\\train_data.csv')
val_df = pd.read_csv('dataset\\val_data.csv')
test_df = pd.read_csv('dataset\\test_data.csv')

X_train = train_df["text"]
y_train = train_df["label"]

X_val = val_df["text"]
y_val = val_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]


vectorizer = TfidfVectorizer(
    max_features=35000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    solver = "saga",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

val_pred = model.predict(X_val_tfidf)

print("Validation Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))

test_pred = model.predict(X_test_tfidf)

print("Test Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

