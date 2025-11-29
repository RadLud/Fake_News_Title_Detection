import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from wordcloud import WordCloud
import re
import os
from nltk.corpus import stopwords
from settings import ADDITIONAL_STOPWORDS, DATA_PATH
from utils import apply_preprocessing, preprocess_text

STOPWORDS = set(stopwords.words("english")) | ADDITIONAL_STOPWORDS

PLOT_DIR = "interpretation_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# 1. TOP WORDS (TF-IDF + model coefficients)

def get_top_words(model, vectorizer, n=30):
    """Return top positive (FAKE) and negative (REAL) words."""
    feature_names = np.array(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
    else:
        # LinearSVC stores coef differently
        coefs = model.coef_.toarray()[0]

    top_fake_idx = np.argsort(coefs)[-n:]
    top_real_idx = np.argsort(coefs)[:n]

    return feature_names[top_fake_idx], feature_names[top_real_idx]


def save_top_words_plot(fake_words, real_words):
    plt.figure(figsize=(10, 6))
    plt.title("Top words: FAKE (red) vs REAL (blue)")
    plt.barh(fake_words, np.arange(len(fake_words)), color="red")
    plt.barh(real_words, -np.arange(len(real_words)), color="blue")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/top_words.png")
    plt.close()

# 2. WORDCLOUD

def generate_wordcloud(texts, filename):
    text = " ".join(texts)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    wc = WordCloud(width=1200, height=500, stopwords=STOPWORDS).generate(text)
    wc.to_file(f"{PLOT_DIR}/{filename}")

# 3. SHAP EXPLAINABILITY (Logistic Regression)

def shap_explain(model, vectorizer, sample_text):
    """Generate SHAP explanation for a single prediction."""

    masker = shap.maskers.Independent(vectorizer.transform([sample_text]))
    explainer = shap.LinearExplainer(
        model, 
        masker=masker,)

    vec = vectorizer.transform([sample_text])
    shap_values = explainer(vec)

    plt.title("SHAP Explanation")
    shap.plots.waterfall(shap_values[0], max_display=20)
    plt.savefig(f"{PLOT_DIR}/shap_explanation.png")
    plt.close()

# 4. WRONG PREDICTIONS

def save_wrong_predictions(model, vectorizer, df):
    X = vectorizer.transform(df["title_clean"])
    preds = model.predict(X)

    df_errors = df[df["label"] != preds]
    df_errors.to_csv(f"{PLOT_DIR}/wrong_predictions.csv", index=False)
    print(f"Saved wrong predictions to {PLOT_DIR}/wrong_predictions.csv")

# 5. main interpretation function
def run_single_interpretation(model_path, tfidf_path, data_path):
    print("\nLoading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(tfidf_path)

    df = pd.read_csv(data_path)
    df= apply_preprocessing(df, col="title")

    print("\nGenerating TOP WORDS...")
    fake_words, real_words = get_top_words(model, vectorizer, n=25)
    save_top_words_plot(fake_words, real_words)

    print("\nGenerating WORDCLOUDS...")
    generate_wordcloud(df[df["label"] == 1]["title_clean"], "wordcloud_fake.png")
    generate_wordcloud(df[df["label"] == 0]["title_clean"], "wordcloud_real.png")

    print("\nSaving WRONG PREDICTIONS...")
    save_wrong_predictions(model, vectorizer, df)

    # Only SHAP for Logistic Regression
    if model.__class__.__name__ == "LogisticRegression":
        print("\nGenerating SHAP example for single text...")
        example = df.iloc[0]["title_clean"]
        shap_explain(model, vectorizer, example)

    print("\nInterpretation complete. Outputs saved to:", PLOT_DIR)

def run_interpretation():
    #run_single_interpretation('models/optuna_best_lr.pkl', 'models/optuna_best_lr_tfidf.pkl', DATA_PATH)
    run_single_interpretation('models/optuna_best_svm.pkl', 'models/optuna_best_svm_tfidf.pkl', DATA_PATH)
