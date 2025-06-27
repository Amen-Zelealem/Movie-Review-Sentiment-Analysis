import sys
import joblib

def load_model(path="model.joblib"):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict_sentiment(model, text):
    prob = model.predict_proba([text])[0]
    pred_label = "positive" if prob[1] > 0.5 else "negative"
    confidence = round(max(prob) * 100, 2)
    return pred_label, confidence

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<review text>\"")
        sys.exit(1)

    review = sys.argv[1].strip()
    if not review:
        print("Error: Review text cannot be empty.")
        sys.exit(1)

    model = load_model()
    label, conf = predict_sentiment(model, review)
    print(f"Prediction: {label} (Confidence: {conf}%)")

if __name__ == "__main__":
    main()
