import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

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

# Label helper
def get_label(label, score):
    if label == "LABEL_1":
        return "Positive", f"😊 Positive Review! (Confidence: {score*100:.1f}%)", "success"
    else:
        return "Negative", f"😞 Negative Review! (Confidence: {score*100:.1f}%)", "error"

# ── Header ──────────────────────────────────────────────────────────────
st.title("🎬 Movie Review Sentiment Analyser")
st.write("Powered by **DistilBERT** fine-tuned on **8,000 IMDB reviews** (T4 GPU)")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📝 Single Review", "📊 Batch Analysis", "ℹ️ About"])

# ── Tab 1: Single Review ──────────────────────────────────────────────────
with tab1:
    st.subheader("Analyse a Single Review")
    user_input = st.text_area("Enter your movie review here:", height=150,
                               placeholder="e.g. This movie was absolutely amazing!")

    if st.button("Analyse Sentiment", use_container_width=True, key="single"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review first!")
        else:
            with st.spinner("Analysing..."):
                result = sentiment(user_input)[0]
                label_name, message, msg_type = get_label(result['label'], result['score'])

                if msg_type == "success":
                    st.success(message)
                else:
                    st.error(message)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", label_name)
                with col2:
                    st.metric("Confidence", f"{result['score']*100:.1f}%")

                st.progress(result['score'])

# ── Tab 2: Batch Analysis ────────────────────────────────────────────────
with tab2:
    st.subheader("Analyse Multiple Reviews at Once")
    st.caption("Enter one review per line")
    batch_input = st.text_area("Enter multiple reviews:", height=200,
                                placeholder="The movie was great!\nTerrible film, waste of time.\nIt was okay, nothing special.")

    if st.button("Analyse All", use_container_width=True, key="batch"):
        if batch_input.strip() == "":
            st.warning("⚠️ Please enter reviews first!")
        else:
            reviews = [r.strip() for r in batch_input.split('\n') if r.strip()]
            with st.spinner(f"Analysing {len(reviews)} reviews..."):
                results = sentiment(reviews)

                positive = 0
                negative = 0
                rows = []

                for review, result in zip(reviews, results):
                    label_name, _, _ = get_label(result['label'], result['score'])
                    rows.append({
                        "Review": review[:100] + ("..." if len(review) > 100 else ""),
                        "Sentiment": label_name,
                        "Confidence": f"{result['score']*100:.1f}%"
                    })
                    if label_name == "Positive":
                        positive += 1
                    else:
                        negative += 1

                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reviews", len(reviews))
                with col2:
                    st.metric("Positive 😊", positive)
                with col3:
                    st.metric("Negative 😞", negative)

                fig = px.pie(
                    values=[positive, negative],
                    names=["Positive", "Negative"],
                    color=["Positive", "Negative"],
                    color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
                    title="Sentiment Breakdown"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Results Table")
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

# ── Tab 3: About ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("About This Project")
    st.markdown("""
    ### 🤖 Model
    - **Architecture**: DistilBERT (distilbert-base-uncased)
    - **Task**: Binary Sentiment Classification (Positive / Negative)
    - **Dataset**: IMDB Movie Reviews (stanfordnlp/imdb)

    ### 📈 Training Details
    | Setting | Value |
    |---|---|
    | Training samples | 8,000 |
    | Test samples | 8,000 |
    | Epochs | 3 |
    | Batch size | 16 |
    | Max token length | 128 |
    | Warmup steps | 150 |
    | Weight decay | 0.01 |
    | Hardware | T4 GPU (Google Colab) |

    ### 🛠️ Tech Stack
    - Python · PyTorch · Hugging Face Transformers
    - Streamlit · Plotly · scikit-learn
    """)
