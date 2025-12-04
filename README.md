📰 Fake News Detection using NLP & Machine Learning
Fake News Detection model built using Python, NLP, TF-IDF, and Logistic Regression.
The model classifies a news article as REAL or FAKE with 90%+ accuracy.

🚀 Features

Text Preprocessing (cleaning, stopword removal, stemming)
TF-IDF vectorization (5000 features)
Logistic Regression classifier
Accuracy: ~90–96%
CLI script for predicting FAKE or REAL news
Clean & modular training + prediction scripts
Easy to extend (LSTM, BERT, Streamlit UI)

📂 Project Structure

    fake-news-detection/
    ├── data/
    │ ├── True.csv
    │ └── Fake.csv
    ├── models/
    │ ├── fake_news_model.pkl
    │ └── tfidf_vectorizer.pkl  
    ├── train.py
    ├── predict.py
    └── README.md

📊 Dataset
Dataset used: Fake and Real News Dataset – Kaggle
Total: 44,000+ news articles
Balanced classes:
FAKE = 0
REAL = 1
Files: Fake.csv, True.csv

Kaggle Dataset Name:
Fake and Real News Dataset (by Clément Bisaillon)

⚙️ Installation
pip install pandas numpy scikit-learn nltk joblib

🧠 Train the Model
python train.py
This will:
Load & merge dataset
Preprocess text
Vectorize using TF-IDF
Train Logistic Regression
Evaluate model
Save model + vectorizer in /models/

🔍 Predict Fake / Real News
python predict.py
Example:
Enter news article:
The government announced new measures to boost employment...
Prediction: REAL NEWS 👍

🧱 Model Details
Algorithm: Logistic Regression
Vectorizer: TF-IDF
Features: 5000
Preprocessing:
Lowercasing
Punctuation removal
Stopword removal
Stemming (Porter Stemmer)

📈 Future Improvements
Add Streamlit Web Interface
Use LSTM, Bi-LSTM
Fine-tune BERT / DistilBERT
Deploy online
Add EDA visualizations

👨‍💻 Author

Vastani Yash
GitHub: https://github.com/vastani001
