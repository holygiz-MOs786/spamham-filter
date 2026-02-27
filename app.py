import streamlit as st
import pickle

# Load trained model & vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Spam vs Ham Detector", page_icon="📧")

st.title("📧 Spam / Ham Message Detector")
st.write("Type a message below and check whether it is **SPAM** or **HAM**.")

# User input
msg = st.text_area("Enter your message:")

# Predict button
if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        msg_vec = vectorizer.transform([msg])

        # Probability-based prediction
        prob = model.predict_proba(msg_vec)[0][1]

        if prob > 0.3:
            st.error(f"🚨 SPAM Message\n\nConfidence: {prob:.2f}")
        else:
            st.success(f"✅ HAM Message\n\nConfidence: {1 - prob:.2f}")
