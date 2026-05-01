import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="Sentiment Analyser",
    page_icon="🎬",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="./my_sentiment_model",
        tokenizer="./my_sentiment_model"
    )

sentiment = load_model()

# UI
st.title("🎬 Movie Review Sentiment Analyser")
st.write("Powered by DistilBERT fine-tuned on IMDB dataset")

st.divider()

# Single review analysis
st.subheader("📝 Analyse a Review")
user_input = st.text_area("Enter your movie review here:", height=150)

if st.button("Analyse Sentiment", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("Analysing..."):
            result = sentiment(user_input)[0]
            label = result['label']
            score = result['score']

            if label == "LABEL_1":
                st.success(f"😊 Positive Review! (Confidence: {score*100:.1f}%)")
            else:
                st.error(f"😞 Negative Review! (Confidence: {score*100:.1f}%)")

st.divider()

# Batch analysis
st.subheader("📊 Analyse Multiple Reviews")
batch_input = st.text_area("Enter multiple reviews:", height=200)

if st.button("Analyse All", use_container_width=True):
    if batch_input.strip() == "":
        st.warning("⚠️ Please enter reviews first!")
    else:
        reviews = [r.strip() for r in batch_input.split('\n') if r.strip()]
        with st.spinner("Analysing all reviews..."):
            results = sentiment(reviews)

            positive = 0
            negative = 0

            for review, result in zip(reviews, results):
                label = result['label']
                score = result['score']
                if label == "LABEL_1":
                    st.success(f"😊 **Positive** ({score*100:.1f}%) — {review[:80]}...")
                    positive += 1
                else:
                    st.error(f"😞 **Negative** ({score*100:.1f}%) — {review[:80]}...")
                    negative += 1

            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Reviews", len(reviews))
            with col2:
                st.metric("Positive 😊", positive)
            with col3:
                st.metric("Negative 😞", negative)