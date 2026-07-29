from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_csv("dataset2\\train.csv")
val_df = pd.read_csv("dataset2\\validation.csv")
test_df = pd.read_csv("dataset2\\test.csv")

train_df["cleaned_text"] = train_df["cleaned_text"].fillna("")
val_df["cleaned_text"] = val_df["cleaned_text"].fillna("")
test_df["cleaned_text"] = test_df["cleaned_text"].fillna("")

X_train = train_df["cleaned_text"]
y_train = train_df["target"]

X_val = val_df["cleaned_text"]
y_val = val_df["target"]

X_test = test_df["cleaned_text"]
y_test = test_df["target"]

vectorizer = CountVectorizer(
    max_features=200000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95
)

print("Fitting vectorizer...")
X_train = vectorizer.fit_transform(X_train)

print("Transforming validation...")
X_val = vectorizer.transform(X_val)

print("Transforming test...")
X_test = vectorizer.transform(X_test)


model = MultinomialNB(alpha=2.0)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

print("Validation:", accuracy_score(y_val, val_pred))
print("Test:", accuracy_score(y_test, test_pred))
    
print("Validation Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))
    
test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))