import streamlit as st
import joblib

# load model and vectorizer
model = joblib.load("best_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

# title
st.title("💬 Sentiment Analysis App")
st.write("Enter your text and get sentiment with rating score 🔥")

# input
review = st.text_area("Enter your text 👇", height=150)

# predict
if st.button("Analyze Sentiment 🚀"):

    if review.strip() == "":
        st.warning("⚠️ Please enter text")
    else:
        # vectorize
        review_vector = vectorizer.transform([review])

        # probability
        probs = model.predict_proba(review_vector)[0]

        negative_score = probs[0] * 100
        positive_score = probs[1] * 100

        prediction = model.predict(review_vector)[0]

        # output
        if str(prediction).lower() in ["positive", "pos", "1"]:
            st.success(f"🟢 Positive Sentiment")
            st.metric("Rating Score", f"{positive_score:.2f}% 👍")

        else:
            st.error(f"🔴 Negative Sentiment")
            st.metric("Rating Score", f"{negative_score:.2f}% 👎")
