from predict import load_predictor
from eda import run_eda
from interpretation import run_interpretation
from train_baseline import run_baseline_training
from train_models import run_model_training
from train_optuna import run_optuna

def main():
    run_eda()
    run_baseline_training()
    run_model_training()
    run_optuna()
    run_interpretation()

def predict(headline):
    model = load_predictor()
    result = model.predict(headline)
    print(result)


predict("BREAKING: Obama admits something shocking")
predict("New advancements in AI technology unveiled")