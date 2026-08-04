import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
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
    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
    )

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

# ==========================================================
# Prepare Tagged Documents
# ==========================================================

print("\nPreparing Tagged Documents...")

train_documents = [
    TaggedDocument(
        words=text.split(),
        tags=[str(i)]
    )
    for i, text in enumerate(train_df[TEXT_COLUMN])
]

print("\nTraining Doc2Vec...\n")

doc2vec = Doc2Vec(
    vector_size=400,
    window=10,
    min_count=2,
    workers=os.cpu_count(),
    epochs=60,
    dm=1,
    negative=15,
    sample=1e-5,
    seed=42
)

doc2vec.build_vocab(train_documents)

doc2vec.train(
    train_documents,
    total_examples=doc2vec.corpus_count,
    epochs=doc2vec.epochs
)

print("\nDoc2Vec Training Complete!")

EMBEDDING_DIM = doc2vec.vector_size

print("\nEmbedding Dimension:", EMBEDDING_DIM)


print("\nGenerating Training Embeddings...")

X_train = np.array([
    doc2vec.dv[str(i)]
    for i in tqdm(range(len(train_df)))
])

print("\nGenerating Validation Embeddings...")

X_val = np.array([
    doc2vec.infer_vector(
        text.split(),
        epochs=50
    )
    for text in tqdm(val_df[TEXT_COLUMN])
])

print("\nGenerating Test Embeddings...")

X_test = np.array([
    doc2vec.infer_vector(
        text.split(),
        epochs=20
    )
    for text in tqdm(test_df[TEXT_COLUMN])
])

y_train = train_df[LABEL_COLUMN]
y_val = val_df[LABEL_COLUMN]
y_test = test_df[LABEL_COLUMN]

print("\nTraining Embedding Shape :", X_train.shape)
print("Validation Shape         :", X_val.shape)
print("Test Shape               :", X_test.shape)

# ==========================================================
# Train Logistic Regression
# ==========================================================

print("\nTraining Logistic Regression...\n")

model = LogisticRegression(
    max_iter=10000,
    random_state=42
)

model.fit(X_train, y_train)

print("\nTraining Complete!")

# ==========================================================
# Validation
# ==========================================================

val_pred = model.predict(X_val)

print("\n========== VALIDATION ==========\n")

print("Accuracy :", accuracy_score(y_val, val_pred))
print("Precision:", precision_score(y_val, val_pred))
print("Recall   :", recall_score(y_val, val_pred))
print("F1 Score :", f1_score(y_val, val_pred))

# ==========================================================
# Test
# ==========================================================

test_pred = model.predict(X_test)

print("\n========== TEST ==========\n")

print("Accuracy :", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred))
print("Recall   :", recall_score(y_test, test_pred))
print("F1 Score :", f1_score(y_test, test_pred))

print("\nClassification Report\n")

print(classification_report(y_test, test_pred))

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, test_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

disp.plot(cmap="Blues")
plt.title("Doc2Vec + Logistic Regression")
plt.show()

print("\nTraining Completed!")