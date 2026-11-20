# Credit Card Application Classifier

A local Streamlit app for the trained pipeline from `CC_fraud_detection.ipynb`. It takes the anonymized input fields `A1` through `A14` and returns the model's `Class` prediction and probability scores.

The notebook's saved model is a tuned logistic regression pipeline (`C=0.1`, `max_iter=400`). Its held-out test results were F1 0.835 and ROC AUC 0.924. The dataset and notebook do not establish that `Class` means fraud, so the app reports the target as `Class` without assigning it a fraud interpretation.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The included `model.joblib` is the serialized trained pipeline. No CSV is needed for predictions. The categorical options reflect values found in the supplied training data; numeric inputs accept any number.

## Hosting

This folder is self-contained and can be pushed to a Streamlit Community Cloud app or another host that supports Streamlit. Select `app.py` as the entry point. Keep `model.joblib`, `requirements.txt`, and `app.py` together.
