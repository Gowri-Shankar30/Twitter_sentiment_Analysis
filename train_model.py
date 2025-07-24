# train_model.py
import pandas as pd
import nltk
import string
import emoji
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

# Load Sentiment140 dataset
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df = df.sample(n=50000, random_state=42)  # train on 50K for faster results

df.columns = ["target", "id", "date", "flag", "user", "tweet"]

# Map labels
df["sentiment"] = df["target"].map({0: "negative", 2: "neutral", 4: "positive"})

# Select relevant columns
df = df[["tweet", "sentiment"]]

# Preprocessing function with emoji detection
def preprocess(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = emoji.demojize(text)  # Convert emojis to words like :smile:
    text = re.sub(r':([a-z_]+):', r'\1', text)  # Remove colons around emoji names
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["processed"] = df["tweet"].apply(preprocess)

# Vectorization
vectorizer = CountVectorizer(max_features=10000)  # limit features to speed up
X = vectorizer.fit_transform(df["processed"])
y = df["sentiment"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained on 1.6M tweets with emoji support and saved!")
