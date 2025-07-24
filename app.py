# app.py
import streamlit as st
from model import predict_sentiment
from datetime import datetime
import time

# Page config
st.set_page_config(page_title="Live Twitter Sentiment", layout="centered")

st.title("📡 Real-time Tweet Sentiment Simulator")
st.write("Simulate live Twitter sentiment classification by entering tweets manually.")

# Initialize session state for simulated tweet stream
if "tweet_log" not in st.session_state:
    st.session_state.tweet_log = []

# Input form
with st.form("tweet_form", clear_on_submit=True):
    tweet_input = st.text_area("✍️ Enter a Tweet", height=100)
    submit = st.form_submit_button("Submit Tweet")

# Handle submission
if submit:
    if tweet_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        sentiment = predict_sentiment(tweet_input)
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.tweet_log.insert(0, {
            "tweet": tweet_input.strip(),
            "sentiment": sentiment.capitalize(),
            "time": timestamp
        })

# Stream display
st.markdown("---")
st.subheader("📜 Tweet Sentiment Stream")

if st.session_state.tweet_log:
    for entry in st.session_state.tweet_log:
        if entry["sentiment"] == "Positive":
            color = "🟢"
        elif entry["sentiment"] == "Negative":
            color = "🔴"
        else:
            color = "🟡"
        st.markdown(f"**{entry['time']}** | {color} **{entry['sentiment']}** — {entry['tweet']}")
else:
    st.info("No tweets yet. Enter a tweet above to simulate live streaming.")

# Option to clear session state
if st.button("🧹 Clear Stream"):
    st.session_state.tweet_log = []
