#!/usr/bin/env python3
# Usage:
#   python predict_email.py "Subject: ... Body: ..."
import sys, pickle

MODEL_PATH = "model_phish.pkl"
VEC_PATH = "vectorizer_phish.pkl"

def load_artifacts(model_path=MODEL_PATH, vec_path=VEC_PATH):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_email.py \"<email subject and body>\"")
        sys.exit(1)
    text = sys.argv[1]
    model, vectorizer = load_artifacts()
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    print(f"Prediction: {pred}")

if __name__ == "__main__":
    main()
