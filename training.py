import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

train_df = pd.read_csv("dataset_pre_processed/train.csv")
val_df = pd.read_csv("dataset_pre_processed/validation.csv")
test_df = pd.read_csv("dataset_pre_processed/test.csv")

TEXT_COLUMN = "cleaned_text"
LABEL_COLUMN = "target"

for df in (train_df, val_df, test_df):
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


print("\nLoading Google News Word2Vec...")

word2vec = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz",
    binary=True
)

print("Vocabulary Size:", len(word2vec.index_to_key))

EMBEDDING_DIM = word2vec.vector_size

word_cache = {}

def sentence_vector(sentence):

    words = sentence.split()

    valid_words = []

    for word in words:

        if word not in word_cache:

            if word in word2vec:
                word_cache[word] = word2vec[word]
            else:
                word_cache[word] = None

        if word_cache[word] is not None:
            valid_words.append(word)

    if len(valid_words) == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    vectors = np.vstack([word_cache[word] for word in valid_words])

    return vectors.mean(axis=0)

def build_embeddings(texts, split_name):

    file_name = f"{split_name}_embeddings.npy"

    if os.path.exists(file_name):
        print(f"\nLoading {file_name}...")
        return np.load(file_name)

    print(f"\nCreating {split_name} embeddings...")

    embeddings = np.zeros(
        (len(texts), EMBEDDING_DIM),
        dtype=np.float32
    )

    for i, text in enumerate(tqdm(texts)):
        embeddings[i] = sentence_vector(text)

    np.save(file_name, embeddings)

    print(f"{file_name} saved.")

    return embeddings



X_train = build_embeddings(
    train_df[TEXT_COLUMN].tolist(),
    "train"
)

X_val = build_embeddings(
    val_df[TEXT_COLUMN].tolist(),
    "validation"
)

X_test = build_embeddings(
    test_df[TEXT_COLUMN].tolist(),
    "test"
)

y_train = train_df[LABEL_COLUMN]
y_val = val_df[LABEL_COLUMN]
y_test = test_df[LABEL_COLUMN]

print("\nEmbedding Shape:", X_train.shape)


print("\nTraining Logistic Regression...\n")

model = LogisticRegression(
    max_iter=10000,
    random_state=42
)

model.fit(X_train, y_train)


val_pred = model.predict(X_val)

print("\n========== VALIDATION ==========\n")

print("Accuracy :", accuracy_score(y_val, val_pred))
print("Precision:", precision_score(y_val, val_pred))
print("Recall   :", recall_score(y_val, val_pred))
print("F1 Score :", f1_score(y_val, val_pred))


test_pred = model.predict(X_test)

print("\n========== TEST ==========\n")

print("Accuracy :", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred))
print("Recall   :", recall_score(y_test, test_pred))
print("F1 Score :", f1_score(y_test, test_pred))

print("\nClassification Report\n")

print(classification_report(y_test, test_pred))


cm = confusion_matrix(y_test, test_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

disp.plot(cmap="Blues")
plt.title("Word2Vec + Logistic Regression")
plt.show()


joblib.dump(model, "word2vec_logistic_regression.pkl")

print("\nModel saved successfully!")