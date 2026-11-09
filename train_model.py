import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "student_data.csv"
MODEL_FILE = "student_model.pkl"

def train_and_save():
    print("Downloading UCI Student Performance dataset...")
    student = fetch_ucirepo(id=320)

    df = pd.concat(
        [student.data.features, student.data.targets],
        axis=1
    )

    # If the repository data identifies the course, keep Mathematics records.
    course_col = next((c for c in df.columns if c.lower() in ("course", "subject")), None)
    if course_col:
        math_rows = df[course_col].astype(str).str.lower().isin(
            ["math", "mat", "mathematics", "maths"]
        )
        if math_rows.any():
            df = df.loc[math_rows].copy()

    if "G3" not in df.columns:
        raise ValueError("The downloaded dataset does not contain the required G3 target column.")

    df = df.drop_duplicates().copy()
    df.to_csv(DATA_FILE, index=False)

    print("Rows and columns:", df.shape)
    print(df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())

    # 1 = at risk (G3 < 10); 0 = not at risk
    y = (df["G3"] < 10).astype(int)

    # G3 must never be used as an input feature.
    X = df.drop(columns=["G3"])
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions,
        target_names=["Not at risk", "At risk"],
        zero_division=0
    )

    print("\nAccuracy:", round(accuracy, 3))
    print("\nClassification report:\n", report)

    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved {MODEL_FILE} and {DATA_FILE}")
    return model, df, accuracy, report

if __name__ == "__main__":
    train_and_save()
