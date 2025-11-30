import joblib
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords")

MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

def predict_news(news):
    cleaned = clean_text(news)
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)[0]
    return "REAL NEWS 👍" if pred == 1 else "FAKE NEWS 🚫"

print("===== Fake News Detection =====")

while True:
    text = input("\nEnter news article (or type exit): ")
    if text.lower() == "exit":
        break
    print("Prediction:", predict_news(text))
