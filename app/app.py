from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.joblib')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    prob = model.predict_proba([text])[0]
    prediction = "positive" if prob[1] > 0.5 else "negative"
    confidence = round(max(prob) * 100, 2)

    return render_template(
        'result.html',
        text=text,
        prediction=prediction,
        confidence=confidence
    )

if __name__ == '__main__':
    app.run(debug=True)
