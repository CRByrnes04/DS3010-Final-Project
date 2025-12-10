import gradio as gr
import joblib

model = joblib.load('trained_model.sav')

def predict_function(input_data):
        prediction = model.predict([input_data])[0] # Example for a single prediction
        return f"Prediction: {prediction}"

demo = gr.Interface(
    fn=predict_function,
    inputs="Test",  # Example: text input
    outputs="text", # Example: text output
    title="My ML Model GUI"
)

demo.launch()

