import streamlit as st, pickle
import re

def check_obvious_scams(text):
    scam_patterns = [
        r'bitcoin.*investment',
        r'make.*\$\d+.*daily',
        r'elon musk.*endorses',
        r'send.*bitcoin.*get.*back',
        r'won.*\$\d+.*lottery',
        r'processing fee.*claim prize'
    ]
    
    text_lower = text.lower()
    for pattern in scam_patterns:
        if re.search(pattern, text_lower):
            return "phishing"
    return None

@st.cache_resource
def load_artifacts():
    with open("model_phish.pkl","rb") as f:
        model = pickle.load(f)
    with open("vectorizer_phish.pkl","rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

st.title("AI-Based Phishing Detection (Prototype)")
text = st.text_area("Paste email subject/body:", height=200, placeholder="Subject: ...\nBody: ...")
if st.button("Classify"):
    if not text.strip():
        st.warning("Please paste some text.")
    else:
        # Check for obvious scams first
        scam_check = check_obvious_scams(text)
        if scam_check:
            st.error(f"Prediction: {scam_check.capitalize()}")
            st.caption("Detected by keyword patterns")
        else:
            model, vectorizer = load_artifacts()
            X = vectorizer.transform([text])
            pred = model.predict(X)[0]
            st.success(f"Prediction: {pred.capitalize()}")
        st.caption("Educational prototype only.")