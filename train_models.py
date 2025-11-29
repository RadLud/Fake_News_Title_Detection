# train_models.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from utils import apply_preprocessing, evaluate_model
from settings import TEST_SIZE, MAX_FEATURES, NGRAM_RANGE, MIN_DF, LR_MAX_ITER, DATA_PATH, RANDOM_STATE





def run_model_training():

    # 1. Load dataset
    DATA_PATH = "data/clean_dataset.csv"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run EDA first.")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df = apply_preprocessing(df, col="title")
    X = df["title_clean"]
    y = df["label"]


    # 2. Train-test split (validation used only for tuning later)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )


    # 3. TF-IDF Vectorization
    print("Vectorizing text (TF-IDF)...")
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)


    # 4. Train all models

    results = []

    # 4.1 Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=LR_MAX_ITER, n_jobs=1)
    lr.fit(X_train_tfidf, y_train)
    results.append(evaluate_model("Logistic Regression", lr, X_test_tfidf, y_test))

    # 4.2 Linear SVM
    print("Training Linear SVM...")
    svm = LinearSVC()
    svm.fit(X_train_tfidf, y_train)
    results.append(evaluate_model("Linear SVM", svm, X_test_tfidf, y_test))

    # 4.3 Naive Bayes
    print("Training Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    results.append(evaluate_model("Multinomial NB", nb, X_test_tfidf, y_test))

    # 4.4 RandomForest (optional)
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        max_features=0.1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train_tfidf, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test_tfidf, y_test))


    # 5. Save results

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_comparison.csv", index=False)

    print("\nMODEL COMPARISON:")
    print(results_df)

    # 5. Save best model (optional)
    best = results_df.sort_values("f1_score", ascending=False).iloc[0]
    print("\nBest model:", best["model"])

    joblib.dump(lr, "models/logreg_tfidf.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

    print("\nDone.")
