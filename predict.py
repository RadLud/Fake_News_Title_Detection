import joblib
from utils import preprocess_text


class FakeNewsPredictor:
    """Loads trained model + vectorizer and makes predictions on new text."""

    def __init__(self, model_path: str, vectorizer_path: str):
        print(f"Loading model from {model_path}")
        print(f"Loading vectorizer from {vectorizer_path}")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str):
        """Return prediction (0 = real, 1 = fake) and probability."""
        
        cleaned = preprocess_text(text)
        vec = self.vectorizer.transform([cleaned])

        label = int(self.model.predict(vec)[0])

        # probability only if model supports predict_proba
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(vec)[0][label])
        else:
            prob = None

        return {
            "input": text,
            "cleaned": cleaned,
            "prediction": label,
            "is_fake": bool(label),
            "probability": prob
        }


def load_predictor():
    MODEL = "models/optuna_best_lr.pkl"
    VECT  = "models/optuna_best_lr_tfidf.pkl"
    return FakeNewsPredictor(MODEL, VECT)