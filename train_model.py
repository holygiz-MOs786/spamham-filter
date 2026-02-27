import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
# Load dataset
df = pd.read_csv("D:/sem3/ml platform & tools/streamlit/spam_ham/spam.csv")
df = df[['v1', 'v2']]
df.columns = ['target', 'text']
# Encode labels
le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])
X = df['text']
y = df['target']
# Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)
X_vec = vectorizer.fit_transform(X)
# Train model
model = LogisticRegression()
model.fit(X_vec, y)
# Save model & vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
while True:
    msg = input("Enter message (or type exit): ")
    if msg.lower() == "exit":
        break
    # Vectorize input
    msg_vec = vectorizer.transform([msg])
    # Get spam probability (class = 1)
    prob = model.predict_proba(msg_vec)[0][1]
    # Threshold-based decision
    if prob > 0.3:
        print(f"Prediction: SPAM  (confidence: {prob:.2f})")
    else:
        print(f"Prediction: HAM   (confidence: {1-prob:.2f})")

    

