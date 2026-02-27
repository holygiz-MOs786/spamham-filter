import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

@st.cache_resource
def train_model():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['target', 'text']
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(df['text'])
    model = LogisticRegression()
    model.fit(X_vec, df['target'])
    return model, vectorizer

model, vectorizer = train_model()

st.set_page_config(page_title="Spam vs Ham Detector", page_icon="📧")
st.title("📧 Spam / Ham Message Detector")
st.write("Type a message below and check whether it is **SPAM** or **HAM**.")

msg = st.text_area("Enter your message:")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        msg_vec = vectorizer.transform([msg])
        prob = model.predict_proba(msg_vec)[0][1]
        if prob > 0.3:
            st.error(f"🚨 SPAM Message\n\nConfidence: {prob:.2f}")
        else:
            st.success(f"✅ HAM Message\n\nConfidence: {1 - prob:.2f}")
