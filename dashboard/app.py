import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG (DARK MODE READY) ----------------
st.set_page_config(
    page_title="Emotion AI Pro",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("../models/emotion_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_prob" not in st.session_state:
    st.session_state.last_prob = None

# ---------------- UI ----------------
st.title("🧠 Emotion Detection AI PRO")
st.write("Advanced NLP Emotion Classification System")

# ---------------- INPUT ----------------
user_input = st.text_area("Enter your text")

# ---------------- PREDICT ----------------
if st.button("Predict Emotion"):

    if user_input.strip() != "":

        vec = vectorizer.transform([user_input])

        prediction = model.predict(vec)[0]

        # Confidence score (if model supports predict_proba)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)
            confidence = np.max(proba)
            st.session_state.last_prob = proba[0]
        else:
            confidence = None

        st.session_state.history.append(prediction)

        st.success(f"Emotion: **{prediction}**")

        if confidence:
            st.info(f"Confidence: {confidence:.2f}")

    else:
        st.warning("Please enter text")

# ---------------- EXAMPLE ----------------
st.markdown("---")
st.subheader("Try Example")

if st.button("Example Text"):

    sample = "I am very happy and excited today"

    vec = vectorizer.transform([sample])
    prediction = model.predict(vec)[0]

    st.session_state.history.append(prediction)

    st.info(sample)
    st.success(f"Emotion: **{prediction}**")

# ---------------- PIE CHART ----------------
st.markdown("---")
st.subheader("Emotion Analytics")

if len(st.session_state.history) > 0:

    df = pd.DataFrame(st.session_state.history, columns=["emotion"])

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df["emotion"].value_counts())

    with col2:
        st.write("Pie Chart")
        pie_data = df["emotion"].value_counts()
        st.pyplot(pie_data.plot.pie(autopct="%1.1f%%").figure)

else:
    st.info("No predictions yet")