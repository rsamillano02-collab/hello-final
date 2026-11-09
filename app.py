
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Wine Quality Prediction System",
    page_icon="🍷",
    layout="wide"
)


# ---------------------------------
# CUSTOM DESIGN
# ---------------------------------

st.markdown("""
<style>
.main {
    background-color: #f8f9fc;
}
h1 {
    color: #7b1e3a;
}
h2, h3 {
    color: #51202f;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #eadde1;
}
.stButton button {
    background-color: #7b1e3a;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------
# DATASET SOURCE
# ---------------------------------

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/"
    "winequality-red.csv"
)

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]


# ---------------------------------
# LOAD DATASET
# ---------------------------------

@st.cache_data
def load_data():
    if os.path.exists("wine_quality.csv"):
        df = pd.read_csv("wine_quality.csv")
    else:
        df = pd.read_csv(DATA_URL, sep=";")

    return df


# ---------------------------------
# TRAIN OR LOAD MODEL
# ---------------------------------

@st.cache_resource
def load_model():

    if os.path.exists("wine_model.pkl"):
        return joblib.load("wine_model.pkl")

    df = load_data()
    df = df.drop_duplicates().copy()

    y = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality"])

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X, y)

    return model


# ---------------------------------
# LOAD APPLICATION RESOURCES
# ---------------------------------

try:
    data = load_data()
    model = load_model()

except Exception as e:
    st.error("Unable to load the dataset or prediction model.")
    st.error(str(e))
    st.info("Please check your internet connection and project files.")
    st.stop()


# ---------------------------------
# HEADER
# ---------------------------------

st.title("🍷 Wine Quality Prediction System")

st.markdown("""
### Welcome to the Wine Quality Prediction Dashboard

This system uses Machine Learning to classify red wine samples
based on their physicochemical measurements.

It classifies wine samples into:

- GOOD QUALITY
- NEEDS IMPROVEMENT
""")

st.divider()


# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title("🍷 Wine Predictor")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset Preview",
        "Wine Prediction",
        "About System"
    ]
)


# ---------------------------------
# DASHBOARD
# ---------------------------------

if page == "Dashboard":

    st.header("Dashboard Overview")

    total_records = len(data)
    average_quality = round(data["quality"].mean(), 2)
    good_records = int((data["quality"] >= 6).sum())
    needs_improvement = int((data["quality"] < 6).sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", total_records)
    col2.metric("Average Quality", average_quality)
    col3.metric("Good Quality", good_records)
    col4.metric("Needs Improvement", needs_improvement)

    st.subheader("Wine Quality Distribution")

    distribution = (
        data["quality"]
        .apply(lambda x: "GOOD QUALITY" if x >= 6
               else "NEEDS IMPROVEMENT")
        .value_counts()
    )

    st.bar_chart(distribution)

    st.subheader("Dataset Information")

    st.write(
        "The dataset contains physicochemical measurements "
        "of red wine samples from the UCI Machine Learning Repository."
    )

    st.caption("Prediction rule: Quality ≥ 6 = GOOD QUALITY")


# ---------------------------------
# DATASET PREVIEW
# ---------------------------------

elif page == "Dataset Preview":

    st.header("Dataset Preview")

    st.write("Explore the red wine dataset used by the system.")

    st.dataframe(data.head(20), use_container_width=True)

    st.subheader("Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Dataset Shape")
        st.write(data.shape)

    with col2:
        st.write("Missing Values")
        st.write(int(data.isnull().sum().sum()))

    st.subheader("Statistical Description")

    st.dataframe(data.describe(), use_container_width=True)


# ---------------------------------
# WINE PREDICTION
# ---------------------------------

elif page == "Wine Prediction":

    st.header("Predict Wine Quality")

    st.write("""
    Upload a CSV file containing the 11 required wine measurement
    columns. The system will predict the quality category of each sample.
    """)

    # Download sample template

    template = data[FEATURES].head(5)

    st.download_button(
        label="Download Sample Input CSV",
        data=template.to_csv(index=False),
        file_name="sample_wine_input.csv",
        mime="text/csv"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Wine Sample CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            new_data = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")
            st.dataframe(new_data, use_container_width=True)

            missing_columns = [
                col for col in FEATURES
                if col not in new_data.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            elif new_data.empty:

                st.warning("The uploaded CSV contains no records.")

            else:

                input_data = new_data[FEATURES].copy()

                # Check numeric values
                for col in FEATURES:
                    input_data[col] = pd.to_numeric(
                        input_data[col],
                        errors="coerce"
                    )

                if input_data.isnull().all(axis=1).any():
                    st.error(
                        "Some rows have no valid measurement values. "
                        "Please check your CSV."
                    )

                else:

                    predictions = model.predict(input_data)

                    results = new_data.copy()

                    results["Prediction"] = [
                        "GOOD QUALITY" if value == 1
                        else "NEEDS IMPROVEMENT"
                        for value in predictions
                    ]

                    st.success("Prediction completed successfully!")

                    st.subheader("Prediction Results")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    good_count = int((predictions == 1).sum())
                    improvement_count = int((predictions == 0).sum())

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Good Quality Samples",
                        good_count
                    )

                    col2.metric(
                        "Needs Improvement Samples",
                        improvement_count
                    )

                    st.subheader("Prediction Summary")

                    summary = pd.Series(
                        predictions
                    ).map({
                        1: "GOOD QUALITY",
                        0: "NEEDS IMPROVEMENT"
                    }).value_counts()

                    st.bar_chart(summary)

                    # Download results

                    csv_results = results.to_csv(index=False)

                    st.download_button(
                        label="Download Prediction Results",
                        data=csv_results,
                        file_name="wine_quality_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error("Error processing the uploaded CSV.")
            st.error(str(e))


# ---------------------------------
# ABOUT SYSTEM
# ---------------------------------

elif page == "About System":

    st.header("About the System")

    st.write("""
    The Wine Quality Prediction System is a classroom
    decision-support prototype that uses machine learning
    to classify red wine samples.

    **Technologies Used:**

    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Random Forest Classifier
    - Joblib

    **Dataset:**

    UCI Machine Learning Repository - Wine Quality Dataset.

    **Prediction Rule:**

    - Quality score greater than or equal to 6:
      GOOD QUALITY
    - Quality score below 6:
      NEEDS IMPROVEMENT
    """)

    st.warning("""
    This application is intended for preliminary screening only.
    Its predictions are not a replacement for laboratory testing,
    sensory evaluation, or professional quality-control decisions.
    """)

    st.markdown(
        "[Visit UCI Wine Quality Dataset]"
        "(https://archive.ics.uci.edu/dataset/186/wine+quality)"
    )


# ---------------------------------
# FOOTER
# ---------------------------------

st.divider()

st.caption(
    "Wine Quality Prediction System | Data Mining Laboratory | Section B"
)
