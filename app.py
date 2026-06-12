import streamlit as st
import joblib

model = joblib.load("best_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Sentiment Analysis App")

review = st.text_area("Enter Review")

if st.button("Predict"):

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)

    if prediction[0].lower() == "positive":
        st.success(f"Predicted Sentiment: {prediction[0]}")
    else:
        st.error(f"Predicted Sentiment: {prediction[0]}")