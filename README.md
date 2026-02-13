Machine Learning Project (NLP) – Postgraduate Studies (AI)

Author: Radosław

---------------------------------------------

Project Goal

The objective of this project is to build a machine learning model capable of classifying news headlines as either:

- Fake (1)
- Real (0)

The task is performed purely on the headline text.  
This makes the problem both interesting and challenging due to limited textual context.

The project uses classical NLP + ML methods (TF-IDF + linear models), following academic requirements.

---------------------------------------------

Project Structure

project/
│   
  ├── main.py                 # Full pipeline orchestration

  ├── eda.py                  # Download datasets for Kaggle and perform Exploratory Data Analysis

  ├── utils.py                # Utilities

  ├── train_baseline.py       # Baseline Logistic Regression

  ├── train_models.py         # Additional ML models

  ├── train_optuna.py         # Hyperparameter tuning

  ├── evaluate.py             # Evaluation metrics

  ├── interpretation.py       # SHAP + WordCloud and explanations

  ├── predict.py              # Model inference (new headline prediction)

  ├── data/
  
   └── clean_dataset.csv      # Cleaned dataset (ignored in git)

  ├── models/                 # Saved ML models (.pkl)

  ├── plots/                  # EDA plots
  
  ├── interpretation_plots/   # Interpretation plots
  
  └── README.md



---------------------------------------------

Technologies Used

Python 3.12  
Pandas, NumPy  
scikit-learn  
Optuna  
NLTK  
Matplotlib, Seaborn, WordCloud  
SHAP  
Joblib  

---------------------------------------------

Preprocessing

- Lowercasing text  
- Removing HTML tags  
- Removing URLs  
- Removing non-alphabetic characters  
- Tokenization  
- Stopword removal  
- Removing extremely short or long titles  
- Removing duplicates  

Final dataset size: ~62k headlines.

---------------------------------------------

Exploratory Data Analysis (EDA)

Includes:

- Class distribution  
- Word frequency comparison  
- WordClouds  
- Title length histograms  
- Outlier removal  

Plots saved in /plots.

---------------------------------------------

Models Trained

Baseline: Logistic Regression  
- TF-IDF (1–2 ngrams)  
- F1 ≈ 0.888  
- AUC ≈ 0.96  

Other models:
- Linear SVM  
- Random Forest  
- Naive Bayes  

Hyperparameter tuning with Optuna:
- LogisticRegression  
- Linear SVM  

Best model:
- SVM F1 ≈ 0.891  
- AUC ≈ 0.963  

---------------------------------------------

Final Model Performance (Test Set)

Accuracy: 0.90  
Precision: 0.89  
Recall: 0.89  
F1-score: 0.888–0.891  
AUC: 0.963  

Confusion matrix, ROC curve, classification report generated automatically.

---------------------------------------------

Interpretation

Includes:

- WordCloud  
- Frequent words  
- SHAP (LinearExplainer + TF-IDF)  
- Wrong predictions exported to CSV  

---------------------------------------------

Running the Full Pipeline

python main.py

---------------------------------------------

Prediction on New Titles

from predict import load_predictor  
model = load_predictor()  
result = model.predict("BREAKING: Obama admits something shocking")  

Output example:

input: "BREAKING: Obama admits something shocking"  
prediction: 1  
is_fake: true  
probability: 0.91  

---------------------------------------------

Notes for Academic Review

Includes:

- Data loading  
- EDA  
- Preprocessing  
- Baseline  
- Multiple models  
- Metrics  
- ROC, confusion matrix  
- Optuna tuning  
- Interpretation  
- No notebooks — modular Python code  

---------------------------------------------

🙋 Radosław 

