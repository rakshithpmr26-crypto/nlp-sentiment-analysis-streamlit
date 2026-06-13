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
st.write("Enter text and get sentiment with correct confidence score 🔥")

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

        # correct class mapping (IMPORTANT FIX 🔥)
        classes = model.classes_

        prob_dict = dict(zip(classes, probs))

        # -------------------------------
        # SAFE EXTRACTION (FIXED LOGIC)
        # -------------------------------
        positive_score = 0
        negative_score = 0

        for label, prob in prob_dict.items():
            label_str = str(label).lower()

            if label_str in ["positive", "pos", "1", "1.0"]:
                positive_score = prob * 100

            elif label_str in ["negative", "neg", "0", "0.0"]:
                negative_score = prob * 100

        # final prediction
        prediction = model.predict(review_vector)[0]

        # -------------------------------
        # OUTPUT
        # -------------------------------
        if str(prediction).lower() in ["positive", "pos", "1"]:
            st.success("🟢 Positive Sentiment")
            st.metric("Positive Score", f"{positive_score:.2f}% 👍")
            st.metric("Negative Score", f"{negative_score:.2f}% 👎")

        else:
            st.error("🔴 Negative Sentiment")
            st.metric("Negative Score", f"{negative_score:.2f}% 👎")
            st.metric("Positive Score", f"{positive_score:.2f}% 👍")
