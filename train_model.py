
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------
# FILE CONFIGURATION
# ---------------------------------

DATA_FILE = "wine_quality.csv"
MODEL_FILE = "wine_model.pkl"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/"
    "winequality-red.csv"
)


# ---------------------------------
# TRAIN AND SAVE MODEL
# ---------------------------------

def train_and_save():

    print("Downloading UCI Wine Quality Dataset...")

    # Retrieve red wine dataset
    df = pd.read_csv(DATA_URL, sep=";")

    # Save dataset
    df.to_csv(DATA_FILE, index=False)

    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 Records:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    # ---------------------------------
    # DATA PREPROCESSING
    # ---------------------------------

    df = df.drop_duplicates().copy()

    # Target:
    # 1 = GOOD QUALITY
    # 0 = NEEDS IMPROVEMENT

    y = (df["quality"] >= 6).astype(int)

    # Remove original quality column
    # to prevent target leakage
    X = df.drop(columns=["quality"])

    print("\nTarget Distribution:")
    print(y.value_counts())

    # ---------------------------------
    # TRAIN AND TEST SPLIT
    # ---------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ---------------------------------
    # MACHINE LEARNING PIPELINE
    # ---------------------------------

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    # ---------------------------------
    # MODEL TRAINING
    # ---------------------------------

    print("\nTraining Random Forest Model...")

    model.fit(X_train, y_train)

    # ---------------------------------
    # MODEL EVALUATION
    # ---------------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nModel Accuracy:", round(accuracy, 3))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "Needs Improvement",
            "Good Quality"
        ],
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # ---------------------------------
    # SAVE TRAINED MODEL
    # ---------------------------------

    joblib.dump(model, MODEL_FILE)

    print("\nModel successfully saved!")
    print("Dataset:", DATA_FILE)
    print("Model:", MODEL_FILE)

    return model, df, accuracy, report


# ---------------------------------
# MAIN PROGRAM
# ---------------------------------

if __name__ == "__main__":

    train_and_save()
