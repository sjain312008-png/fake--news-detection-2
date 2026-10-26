import streamlit as st
import joblib

# Load the trained ML model
model = joblib.load("fake_news_model .pkl")

# Configure the website
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Website heading
st.title("📰 Fake News Detection")
st.write(
    "Enter a news article below to get a prediction "
    "from our Machine Learning model."
)

# Input box
news_text = st.text_area(
    "Paste your news article here:",
    height=220,
    placeholder="Enter a headline or news article..."
)

# Prediction button
if st.button("Check News", type="primary"):
    if news_text.strip():
        prediction = model.predict([news_text])[0]
        probabilities = model.predict_proba([news_text])[0]

        if prediction == 0:
            st.error("Prediction: FAKE NEWS")
            confidence = probabilities[0] * 100
        else:
            st.success("Prediction: REAL NEWS")
            confidence = probabilities[1] * 100

        st.write(f"Model confidence: {confidence:.2f}%")
        st.caption(
            "This is an ML prediction, not verified fact-checking. "
            "The model may make mistakes."
        )
    else:
        st.warning("Please enter some news first.")
