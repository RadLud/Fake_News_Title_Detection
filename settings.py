# CONFIGURATION SETTINGS FOR THE FAKE NEWS CLASSIFICATION PROJECT

# PATH SETTINGS
DATA_PATH = "data/clean_dataset.csv"

# DATASET INFO
KAGGLE_DATASET_NAME = "saurabhshahane/fake-news-classification" #singular csv file
KAGGLE_DATASET_NAME_2 = "clmentbisaillon/fake-and-real-news-dataset" #this ds has 2 csv files

# TEXT PREPROCESSING
MAX_TITLE_LEN = 40
MIN_TITLE_LEN = 3
URL_PATTERN = r'http|www\.|\.com|\.net|\.org|\.io|\.co|wpengine'
ADDITIONAL_STOPWORDS = {
    'said', 'will', 'one', 'two', 'new', 'also', 'just', 
    'like', 'get', 'week', 'year', 'time', 'make', 'says'
}
NO_OF_WORDS=20 #most common words to print
WORD_CLOUD_TOP_N=100 


# MODEL SETUP
LR_MAX_ITER = 2000
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# TF-IDF SETUP
MAX_FEATURES = 50000
NGRAM_RANGE = (1, 2)
MIN_DF = 5

# OPTUNA SETUP
N_TRIALS = 10