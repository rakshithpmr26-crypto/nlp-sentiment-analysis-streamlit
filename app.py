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
st.title("Sentiment Analysis App")
st.write("Enter text to get sentiment prediction with explanation")

# -------------------------------
# INPUT
# -------------------------------
review = st.text_area("Enter your review", height=150)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review")
    else:

        # -------------------------------
        # TRANSFORM INPUT
        # -------------------------------
        review_vector = vectorizer.transform([review])

        # -------------------------------
        # PREDICTION + PROBABILITY
        # -------------------------------
        probs = model.predict_proba(review_vector)[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, probs))

        positive_score = 0
        negative_score = 0

        for label, prob in prob_dict.items():
            label_str = str(label).lower()

            if label_str in ["positive", "pos", "1", "1.0"]:
                positive_score = prob * 100
            elif label_str in ["negative", "neg", "0", "0.0"]:
                negative_score = prob * 100

        prediction = model.predict(review_vector)[0]

        # -------------------------------
        # OUTPUT RESULT
        # -------------------------------
        if str(prediction).lower() in ["positive", "pos", "1"]:
            st.success("Positive Sentiment")
            st.metric("Positive Score", f"{positive_score:.2f}%")
            st.metric("Negative Score", f"{negative_score:.2f}%")
        else:
            st.error("Negative Sentiment")
            st.metric("Negative Score", f"{negative_score:.2f}%")
            st.metric("Positive Score", f"{positive_score:.2f}%")

        # -------------------------------
        # EXPLANATION SECTION (FIXED)
        # -------------------------------
        st.subheader("Why this prediction?")

        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]

        words = review.lower().split()

        positive_words = []
        negative_words = []

        # FAST LOOKUP DICT
        feature_index = {word: i for i, word in enumerate(feature_names)}

        for word in words:
            if word in feature_index:
                idx = feature_index[word]
                weight = coefs[idx]

                if weight > 0:
                    positive_words.append(word)
                elif weight < 0:
                    negative_words.append(word)

        col1, col2 = st.columns(2)

        with col1:
            st.write("Positive words")
            if positive_words:
                st.write(positive_words)
            else:
                st.write("None detected")

        with col2:
            st.write("Negative words")
            if negative_words:
                st.write(negative_words)
            else:
                st.write("None detected")
