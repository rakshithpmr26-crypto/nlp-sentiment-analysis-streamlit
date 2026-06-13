import streamlit as st
import joblib

# -------------------------------
# LOAD MODEL & VECTORIZER
# -------------------------------
model = joblib.load("best_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("💬 Sentiment Analysis App")
st.write("Enter text and get sentiment with accurate confidence score 🔥")

# -------------------------------
# INPUT
# -------------------------------
review = st.text_area("Enter your review 👇", height=150)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Analyze Sentiment 🚀"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:

        # transform input
        review_vector = vectorizer.transform([review])

        # probabilities
        probs = model.predict_proba(review_vector)[0]

        # correct class mapping
        classes = model.classes_

        # map probabilities safely
        prob_dict = dict(zip(classes, probs))

        # handle both string and numeric labels
        positive_score = prob_dict.get("positive", prob_dict.get(1, 0)) * 100
        negative_score = prob_dict.get("negative", prob_dict.get(0, 0)) * 100

        # final prediction
        prediction = model.predict(review_vector)[0]

        # -------------------------------
        # OUTPUT
        # -------------------------------
        if str(prediction).lower() in ["positive", "1", "pos"]:
            st.success("🟢 Positive Sentiment")
            st.metric("Positive Score", f"{positive_score:.2f}% 👍")
            st.metric("Negative Score", f"{negative_score:.2f}% 👎")

        else:
            st.error("🔴 Negative Sentiment")
            st.metric("Negative Score", f"{negative_score:.2f}% 👎")
            st.metric("Positive Score", f"{positive_score:.2f}% 👍")
