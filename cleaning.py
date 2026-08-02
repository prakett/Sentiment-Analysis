import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FILE = r"dataset_pre_processed\preprocessed_dataset.csv"

TEXT_COLUMN = "cleaned_text"
LABEL_COLUMN = "target"

TARGET_SIZE = 200000
RANDOM_STATE = 42

# ==========================================================
# LOAD DATASET
# ==========================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

# ==========================================================
# HANDLE NULL VALUES
# ==========================================================

print("Replacing null values with empty strings...")

# Replace all NaN values with empty strings
df = df.fillna("")

# Ensure text column contains only strings
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

print(f"\nOriginal Dataset Size: {len(df):,}")

print("\nOriginal Class Distribution:")
print(df[LABEL_COLUMN].value_counts())

# ==========================================================
# CREATE BALANCED 10K DATASET
# ==========================================================

num_classes = df[LABEL_COLUMN].nunique()

if TARGET_SIZE % num_classes != 0:
    raise ValueError(
        f"TARGET_SIZE ({TARGET_SIZE}) must be divisible by "
        f"the number of classes ({num_classes})."
    )

samples_per_class = TARGET_SIZE // num_classes

balanced_df = (
    df.groupby(LABEL_COLUMN, group_keys=False)
      .sample(n=samples_per_class, random_state=RANDOM_STATE)
      .reset_index(drop=True)
)

# Shuffle the balanced dataset
balanced_df = balanced_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("\nBalanced Dataset Size:", len(balanced_df))

print("\nBalanced Class Distribution:")
print(balanced_df[LABEL_COLUMN].value_counts())

# Save balanced dataset
balanced_df.to_csv("balanced_10000.csv", index=False)

# ==========================================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

# 70% Train, 30% Temp
train_df, temp_df = train_test_split(
    balanced_df,
    test_size=0.30,
    stratify=balanced_df[LABEL_COLUMN],
    random_state=RANDOM_STATE
)

# Split remaining 30% equally into Validation and Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df[LABEL_COLUMN],
    random_state=RANDOM_STATE
)

# ==========================================================
# SAVE FILES
# ==========================================================

train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n==============================")
print("DATASET SUMMARY")
print("==============================")

print(f"Original Dataset : {len(df):,}")
print(f"Balanced Dataset : {len(balanced_df):,}")
print(f"Training Set     : {len(train_df):,}")
print(f"Validation Set   : {len(val_df):,}")
print(f"Testing Set      : {len(test_df):,}")

print("\nTraining Distribution:")
print(train_df[LABEL_COLUMN].value_counts())

print("\nValidation Distribution:")
print(val_df[LABEL_COLUMN].value_counts())

print("\nTesting Distribution:")
print(test_df[LABEL_COLUMN].value_counts())

print("\nFiles Created:")
print("---------------------------")
print("balanced_10000.csv")
print("train.csv")
print("validation.csv")
print("test.csv")

print("\nDone!")