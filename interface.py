import gradio as gr
import joblib

# Feature columns expected by the trained model
feature_cols = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area'
]

model = joblib.load('trained_model.sav')


def predict_function(*inputs):
    """Format inputs in the expected column order and run the model."""
    raw = list(inputs)

    # Cast numeric fields; leave categoricals as provided.
    numeric_fields = {
        'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount',
        'Loan_Amount_Term', 'Credit_History'
    }
    processed = []
    for name, value in zip(feature_cols, raw):
        if name in numeric_fields:
            try:
                processed.append(float(value))
            except (TypeError, ValueError):
                processed.append(None)
        else:
            processed.append(value)

    prediction = model.predict([processed])[0]
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
    outputs="text",
    title="Loan Eligibility Prediction",
    description="Provide applicant details to predict loan approval status."
)


demo.launch()

