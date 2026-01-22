# preprocess.py
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""Setup logging"""
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

nltk.download('punkt')
nltk.download('stopwords')

import os

"""Define paths relative to this file (app/core/preprocessor.py)"""
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(base_dir, 'data')
csv_path = os.path.join(data_dir, "spotify_millsongdata.csv")

"""Load and sample dataset"""
try:
    df = pd.read_csv(csv_path).sample(5000)
    logging.info("✅ Dataset loaded and sampled: %d rows", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

"""Drop link column and preprocess"""
df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

"""Text cleaning"""
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

logging.info("🧹 Cleaning text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Text cleaned.")

"""Vectorization"""
logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

"""Cosine similarity"""
logging.info("📐 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity matrix generated.")

"""Save everything"""
joblib.dump(df, os.path.join(data_dir, 'df_cleaned.pkl'))
joblib.dump(tfidf_matrix, os.path.join(data_dir, 'tfidf_matrix.pkl'))
joblib.dump(cosine_sim, os.path.join(data_dir, 'cosine_sim.pkl'))
logging.info("💾 Data saved to disk.")

logging.info("✅ Preprocessing complete.")
