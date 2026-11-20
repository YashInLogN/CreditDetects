from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("model.joblib")
FEATURES = [f"A{i}" for i in range(1, 15)]
CATEGORICAL = {"A4": [1, 2, 3], "A5": list(range(1, 15)), "A6": [1, 2, 3, 4, 5, 7, 8, 9], "A12": [1, 2, 3]}
BINARY = {"A1", "A8", "A9", "A11"}

st.set_page_config(page_title="Credit Card fraud detection Classifier", page_icon="💳", layout="centered")
st.title("Credit Card fraud detection Classifier")
st.caption("Enter the anonymized A1–A14 fields used by the trained model.")
st.info(
    "**Column types:** A1, A8, A9, A11 are binary (0/1); "
    "A4, A5, A6, A12 are categorical codes; "
    "A2, A3, A7, A10, A13, A14 are numeric. "
    "All fields are anonymized in the source dataset."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

with st.form("application"):
    values = {}
    columns = st.columns(2)
    for index, feature in enumerate(FEATURES):
        with columns[index % 2]:
            if feature in BINARY:
                values[feature] = st.selectbox(feature, [i for i in range(11)], format_func=lambda x: str(x))
            elif feature in CATEGORICAL:
                values[feature] = st.selectbox(feature, CATEGORICAL[feature])
            else:
                values[feature] = st.number_input(feature, value=0.0, step=1.0, format="%.4f")
    submitted = st.form_submit_button("Classify application", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{name: values[name] for name in FEATURES}], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    st.subheader(f"Predicted Class: {prediction}")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        st.write({f"Class {label} probability": f"{prob:.1%}" for label, prob in zip(model.classes_, probabilities)})
    st.caption("This is a model output for an anonymized dataset label; confirm the label meaning before using it for operational decisions.")
