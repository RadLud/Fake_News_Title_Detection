import os
import kagglehub
import pandas as pd
import re
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


# -------------Dataset download and loading functions ------------------
    
# Unifify column names 
def reshape_df(df):
    #lets have title, text, label  (add more to column mapping when needed)
    column_mapping = { 
        'title' : 'title',
        'text' : 'text',
        'source_file' : 'label'
    }
    # keep only relevant columns
    df = df[[col for col in column_mapping.keys() if col in df.columns]]
    df = df.rename(columns=column_mapping)
    # change fake/real to 1/0
    df['label'] = df['label'].map({'Fake.csv': 1, 'True.csv': 0})   # not univeral, can be extended
    df['label'] = df['label'].astype(int) 
    df['title'] = df['title'].astype(str)
    df['text'] = df['text'].astype(str)
    
    return df

# Download ds and load to pandas dataframe
def download_dataset(kaggle_ds_name): 
    try:
        print ("Downloading dataset:", kaggle_ds_name)
        folder = kagglehub.dataset_download(kaggle_ds_name)
        #list files in the folder
        #print (folder)
        list_of_files = os.listdir(folder)
        if len(list_of_files) == 1:
            full_path = folder + '/' + list_of_files[0]
            df = pd.read_csv(full_path)
            #kaggle ds often have a index column without a name - we can drop it
            df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
        else:
            #combine csv files into one
            dfs = []
            for file in list_of_files:
                if file.endswith('.csv'):
                    full_path = folder + '/' + file
                    df = pd.read_csv(full_path)
                    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
                    df['source_file'] = file
                    dfs.append(df)
            combined_df = pd.concat(dfs, ignore_index=True)
            df = reshape_df(combined_df)
        df['title'] = df['title'].astype(str)
        df['text'] = df['text'].astype(str)
        print(f"Dataset downloaded and loaded: {kaggle_ds_name}, shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception("Error downloading dataset: " + str(e))




#------------Preprocessing functions ------------------

def preprocess_text(text:str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"<.*?>", " ", text)         # remove HTML tags
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)   # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()   # clean excessive whitespace
    return text

def apply_preprocessing(df: pd.DataFrame, col="title") -> pd.DataFrame:
    # Adds a new column '{col}_clean' with preprocessed text.

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataframe.")
    
    df[f"{col}_clean"] = df[col].apply(preprocess_text)
    return df

#--------------Model Functions ------------------

def print_metrics(name, y_true, y_pred, y_prob):
    """Print evaluation metrics."""
    print(f"\n{name} SET RESULTS:")
    print("-" * 40)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"AUC:       {roc_auc_score(y_true, y_prob):.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Some models (LinearSVC) do NOT support predict_proba
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = None

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc": auc
    }

#-------------------optuna related functions ----------------------
