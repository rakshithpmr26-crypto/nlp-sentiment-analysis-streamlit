import streamlit as st
import joblib

# load model and vectorizer
model = joblib.load("best_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# page config
st.set_page_config(page_title="Amazon Sentiment Analysis", page_icon="🛒", layout="centered")

# title
st.title("🛒 Amazon Sentiment Analysis App")
st.write("Analyze Amazon product reviews using Machine Learning 🤖")

# input
review = st.text_area("Enter Amazon Review 👇", height=150)

# predict button
if st.button("Predict Sentiment 🚀"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:
        # transform
        review_vector = vectorizer.transform([review])

        # prediction
        prediction = model.predict(review_vector)[0]

        # output
        if str(prediction).lower() in ["positive", "1", "pos"]:
            st.success("🟢 Positive Review")
        else:
            st.error("🔴 Negative Review")
