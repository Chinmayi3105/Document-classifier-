import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Document Classifier")

text = st.text_area("Enter your document text:")

if st.button("Classify"):
    if text.strip():
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)
        st.success(f"Prediction: {prediction[0]}")
    else:
        st.warning("Please enter some text.")
        