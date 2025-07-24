# model.py
import nltk
import string
import emoji
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r':([a-z_]+):', r'\1', text)
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def predict_sentiment(text):
    processed = preprocess(text)
    vectorized = vectorizer.transform([processed])
    return model.predict(vectorized)[0]
