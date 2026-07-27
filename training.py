from numpy import test
import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


train_df = pd.read_csv('dataset2\\train.csv')
train_df["cleaned_text"] = train_df["cleaned_text"].fillna("")

val_df = pd.read_csv('dataset2\\validation.csv')
val_df["cleaned_text"] = val_df["cleaned_text"].fillna("")


test_df = pd.read_csv('dataset2\\test.csv')
test_df["cleaned_text"] = test_df["cleaned_text"].fillna("")


X_train = train_df["cleaned_text"]
y_train = train_df["target"]

X_val = val_df["cleaned_text"]
y_val = val_df["target"]

X_test = test_df["cleaned_text"]
y_test = test_df["target"]


vectorizer = CountVectorizer(
    max_features=200000, # good with 200k
    ngram_range=(1, 2),
    min_df = 5,
    max_df = 0.95,
   
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=3000,
    solver = "saga",
    random_state=42,
    C = 1,
)

print("Fitting vectorizer...")
X_train = vectorizer.fit_transform(X_train)

print("Transforming validation...")
X_val = vectorizer.transform(X_val)

print("Transforming test...")
X_test = vectorizer.transform(X_test)

print("Training Logistic Regression...")
model.fit(X_train, y_train)

print("Training complete!")

model.fit(X_train_tfidf, y_train)


val_pred = model.predict(X_val_tfidf)

print("Validation Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))

test_pred = model.predict(X_test_tfidf)

print("Test Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

print(model.n_iter_)





import pickle

# Bundle everything together
sentiment_model = {
    "model": model,
    "vectorizer": vectorizer,
    "label_map": {
        0: "Negative",
        1: "Positive"
    },
    "model_info": {
        "algorithm": "Logistic Regression",
        "solver": "saga",
        "C": 1,
        "max_features": 200000,
        "ngram_range": (1, 2),
        "accuracy": 0.7988125
    }
}

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(sentiment_model, f)

print("Model saved successfully!")

