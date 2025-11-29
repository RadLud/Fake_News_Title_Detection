import optuna
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import f1_score

from utils import apply_preprocessing
from settings import VAL_SIZE, RANDOM_STATE, DATA_PATH , N_TRIALS


# ---------------------------------------------------------
def objective_lr(trial, X_train, X_val, y_train, y_val):
    # TF-IDF hyperparams
    max_features = trial.suggest_int("max_features", 20000, 80000)
    min_df = trial.suggest_int("min_df", 1, 5)
    ngram_str = trial.suggest_categorical("ngram", ["1-1", "1-2", "1-3"])
    ngram = tuple(map(int, ngram_str.split("-")))

    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # LR hyperparams
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])

    model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=1000
    )

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_val_tfidf)

    return f1_score(y_val, y_pred)

# ---------------------------------------------------------
def objective_svm(trial, X_train, X_val, y_train, y_val):
    # TF-IDF hyperparams
    max_features = trial.suggest_int("max_features", 20000, 80000)
    min_df = trial.suggest_int("min_df", 1, 5)
    ngram_str = trial.suggest_categorical("ngram", ["1-1", "1-2", "1-3"])
    ngram = tuple(map(int, ngram_str.split("-")))

    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # Linear SVC hyperparams
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)
    loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
    tol = trial.suggest_float("tol", 1e-5, 1e-3, log=True)

    model = LinearSVC(
        C=C,
        loss=loss,
        tol=tol,
        max_iter=2000
    )

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_val_tfidf)

    return f1_score(y_val, y_pred)


# --------------------------main--------------------------------
def optuna_main(model_name):
    if model_name not in ["lr", "svm"]:
        raise ValueError("model_name must be 'lr' or 'svm'")

    print(f"Running Optuna for model: {model_name}")

    
    # 1. Load & preprocess data
    
    df = pd.read_csv(DATA_PATH)
    df = apply_preprocessing(df, col="title")

    X = df["title_clean"]
    y = df["label"]

    # 2. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 3.Select objective
    if model_name == "lr":
        objective = lambda trial: objective_lr(trial, X_train, X_val, y_train, y_val)
    else:
        objective = lambda trial: objective_svm(trial, X_train, X_val, y_train, y_val)

    
    # 4.Run Optuna
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest params:", study.best_params)
    print("Best F1:", study.best_value)
    best = study.best_params
    
    # 5. Retrain final model on full training dataset
    # -------------------------------------
    print("\nTraining final model on training dataset...")
    ngram = tuple(map(int, best["ngram"].split("-")))
    # 5.1 Build TF-IDF
    tfidf = TfidfVectorizer(
        max_features=best["max_features"],
        min_df=best["min_df"],
        ngram_range=ngram
    )
    X_train_tfidf = tfidf.fit_transform(X_train)

    # 5.2 Build final model
    if model_name == "lr":
        model = LogisticRegression(
            C=best["C"],
            solver=best["solver"],
            penalty="l2",
            max_iter=1000
        )
        model_file = "optuna_best_lr"
    else:
        model = LinearSVC(
            C=best["C"],
            loss=best["loss"],
            tol=best["tol"],
            max_iter=2000
        )
        model_file = "optuna_best_svm"

    model.fit(X_train_tfidf, y_train)

    # Save
    joblib.dump(model, f"models/{model_file}.pkl")
    joblib.dump(tfidf, f"models/{model_file}_tfidf.pkl")

    print(f"Saved {model_file} and its TF-IDF vectorizer.")

#-------------------------run--------------------------------
def run_optuna():
    optuna_main(model_name="lr")
    optuna_main(model_name="svm")