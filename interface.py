import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Feature columns expected by the trained model (order matters)
feature_cols = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area'
]

# Simple encoders to turn categoricals into numeric values; adjust to match training if needed.
encoders = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
}

numeric_fields = {
    'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount',
    'Loan_Amount_Term', 'Credit_History', 'Dependents'
}

model = joblib.load('trained_model.sav')


def _coerce_numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def predict_function(*inputs):
    """Format inputs in the expected column order and run the model."""
    processed = []
    for name, value in zip(feature_cols, inputs):
        if name in encoders:
            processed.append(encoders[name].get(value, np.nan))
        elif name == 'Dependents':
            # Dependents comes as string from dropdown; allow numeric conversion
            processed.append(_coerce_numeric(value))
        elif name in numeric_fields:
            processed.append(_coerce_numeric(value))
        else:
            processed.append(value)

    # Create a DataFrame with column names to satisfy models fitted with feature names
    X = pd.DataFrame([processed], columns=feature_cols)
    prediction = model.predict(X)[0]
    return f"Prediction: {prediction}"


demo = gr.Interface(
    fn=predict_function,
    inputs=[
        gr.Dropdown(['Male', 'Female'], label='Gender'),
        gr.Dropdown(['Yes', 'No'], label='Married'),
        gr.Dropdown(['0', '1', '2', '3', '4'], label='Dependents'),
        gr.Dropdown(['Graduate', 'Not Graduate'], label='Education'),
        gr.Dropdown(['Yes', 'No'], label='Self Employed'),
        gr.Number(label='Applicant Income'),
        gr.Number(label='Coapplicant Income'),
        gr.Number(label='Loan Amount'),
        gr.Number(label='Loan Amount Term'),
        gr.Number(label='Credit History'),
        gr.Dropdown(['Urban', 'Semiurban', 'Rural'], label='Property Area'),
    ],
    outputs="Loan Status",
    title="Loan Eligibility Prediction",
    description="Provide applicant details to predict loan approval status."
)


demo.launch()

