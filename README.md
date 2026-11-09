# Student Performance Prediction System

A classroom machine-learning prototype built with Python, scikit-learn, GitHub, and Streamlit.

## Project purpose
The system explores whether historical student information can help flag learners who may need additional academic support.

## Dataset
UCI Machine Learning Repository — Student Performance:
https://archive.ics.uci.edu/dataset/320/student+performance

The project retrieves the dataset using `ucimlrepo` (dataset ID 320).

## Prediction target
- `G3 < 10` = **AT RISK**
- `G3 >= 10` = **NOT AT RISK**

**Important:** `G3` is the target only. It is removed from the input features to prevent target leakage.

## Technologies
- Python
- pandas
- scikit-learn (Random Forest classifier)
- Streamlit
- ucimlrepo
- joblib

## Project files
- `train_model.py` — downloads the dataset, preprocesses data, trains/evaluates the model, and saves the model.
- `app.py` — Streamlit dashboard, CSV upload, predictions, and result download.
- `requirements.txt` — required Python packages.

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the model and download the dataset:

   ```bash
   python train_model.py
   ```

5. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL shown in the terminal (normally `http://localhost:8501`).
7. Download the sample input CSV from the app, then upload it to test predictions.

If `student_model.pkl` or `student_data.csv` is missing, the app attempts to create them automatically. An internet connection is required for the initial dataset download.

## GitHub upload
Create a public repository named `student-performance-prediction`, then upload:
- `app.py`
- `train_model.py`
- `requirements.txt`
- `README.md`

The generated dataset and model files are not required in the repository because the app can generate them when first run. Do not upload passwords, access tokens, or other secrets.

## Limitations and responsible use
This is a classroom demonstration using public historical data. A model trained on one dataset may not generalize to another institution or population. Predictions are not diagnoses and must not be used as the sole basis for real student decisions.
