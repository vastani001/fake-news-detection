import os
import string
import pandas as pd
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords
nltk.download("stopwords")

# Paths
TRUE_PATH = "data/True.csv"
FAKE_PATH = "data/Fake.csv"
MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# Load Dataset
true_df = pd.read_csv(TRUE_PATH)
fake_df = pd.read_csv(FAKE_PATH)

true_df["label"] = 1   # REAL
fake_df["label"] = 0   # FAKE

true_df["content"] = true_df["title"] + " " + true_df["text"]
fake_df["content"] = fake_df["title"] + " " + fake_df["text"]

df = pd.concat([true_df[["content", "label"]], fake_df[["content", "label"]]])
df.dropna(inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

print("Dataset loaded:", df.shape)

# Clean text
df["cleaned"] = df["content"].apply(clean_text)

# Split
X = df["cleaned"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# Evaluate
pred = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

# Save artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(tfidf, VECTORIZER_PATH)

print("\nModel & Vectorizer Saved Successfully!")
