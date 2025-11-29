import os
import pandas as pd
import joblib
from settings import VAL_SIZE, TEST_SIZE, MAX_FEATURES, NGRAM_RANGE, MIN_DF, LR_MAX_ITER, DATA_PATH

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import apply_preprocessing, print_metrics



def run_baseline_training():
    # 1. Load Cleaned Dataset

    print("Loading dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run EDA first.")
    else:
        df = pd.read_csv(DATA_PATH)

    # 2. Preprocessing

    print("Applying preprocessing...")
    df = apply_preprocessing(df, col="title")

    # 3. Train /  Val / Test Split

    X = df["title_clean"]
    y = df["label"]

    print("Performing train/val/test split...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),    
        random_state=42,
        stratify=y_temp
    )
    print(f"Train size: {len(X_train)}")
    print(f"Val size:   {len(X_val)}")
    print(f"Test size:  {len(X_test)}")

    # 4. TF-IDF Vectorization

    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf   = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    # 5. Train Baseline Model (Logistic Regression)

    print("Training Logistic Regression model...")

    model = LogisticRegression(
        max_iter=LR_MAX_ITER,
        n_jobs=1,
        solver="lbfgs"
    )

    model.fit(X_train_tfidf, y_train)

    # 6. Evaluation on validation set

    print("Evaluating model on validation set...")

    y_val_pred = model.predict(X_val_tfidf)
    y_val_prob = model.predict_proba(X_val_tfidf)[:, 1]

    print_metrics("VALIDATION", y_val, y_val_pred, y_val_prob)

    # 7. Evaluation on test set

    print("Evaluating model on test set...")

    y_test_pred = model.predict(X_test_tfidf)
    y_test_prob = model.predict_proba(X_test_tfidf)[:, 1]

    print_metrics("TEST", y_test, y_test_pred, y_test_prob)


    # 7. Save Model + Vectorizer

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/baseline_logreg_tfidf.pkl")
    joblib.dump(tfidf, "models/baseline_tfidf_vectorizer.pkl")

    print("\nSaved model to models/baseline_logreg_tfidf.pkl")
    print("Saved TF-IDF vectorizer to models/baseline_tfidf_vectorizer.pkl")

    print("\nBaseline training complete.")
