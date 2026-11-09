import os
import streamlit as st
import pandas as pd
import joblib
from train_model import train_and_save, MODEL_FILE, DATA_FILE

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    # Create the model and dataset automatically if they are not present.
    if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
        train_and_save()
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_FILE):
        train_and_save()
    return pd.read_csv(DATA_FILE)

st.title("🎓 Student Performance Prediction System")
st.caption("Classroom prototype for exploring possible academic risk. Predictions are not diagnoses.")

try:
    model = load_model()
    data = load_dataset()
except Exception as e:
    st.error("The model or dataset could not be prepared. Check your internet connection and dependencies, then try again.")
    st.exception(e)
    st.stop()

if "G3" not in data.columns:
    st.error("The dataset does not contain the required G3 column.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Student records", len(data))
c2.metric("Average final grade (G3)", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded student records")
st.write("Upload a CSV file containing the same predictor columns used to train the model. Do not include G3.")
template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "⬇️ Download sample input CSV",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="sample_prediction_input.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Choose student CSV file", type=["csv"])

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        required_columns = list(model.feature_names_in_)
        missing = [col for col in required_columns if col not in new_data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            # Ignore extra columns and keep the exact training feature order.
            prediction_input = new_data[required_columns]
            predictions = model.predict(prediction_input)

            results = new_data.copy()
            results["Prediction"] = [
                "AT RISK" if value == 1 else "NOT AT RISK"
                for value in predictions
            ]

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted at-risk records", int((predictions == 1).sum()))

            st.download_button(
                "⬇️ Download prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="student_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error("The uploaded file could not be processed. Please check that it is a valid CSV.")
        st.exception(e)

st.divider()
st.caption("Responsible use: This model is trained on historical data from the UCI Student Performance dataset. It may not generalize to other schools and must not be the sole basis for student decisions.")
