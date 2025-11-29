import settings
import utils
import pandas as pd
import numpy as np

#load datasets
df_1 = pd.DataFrame(utils.download_dataset(settings.KAGGLE_DATASET_NAME))
df_2 = pd.DataFrame(utils.download_dataset(settings.KAGGLE_DATASET_NAME_2))
df = pd.concat([df_1, df_2], ignore_index=True)

print(df.head())
print(df.info())
print(df['label'].value_counts())

# 1. find duplicates by title
dupes = df[df.duplicated(subset=["title"], keep=False)]

# 1.1 duplicate titles with both labels - remove all of them
conflicts = dupes.groupby("title")["label"].nunique()
conflict_titles = conflicts[conflicts == 2].index.tolist()
df = df[~df["title"].isin(conflict_titles)]

# 1.2 remove normal duplicated - keep first
df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

print("Final dataset size:", len(df))
print("Fake vs Real:")
print(df['label'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns


# see the class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df['label'])
plt.title("Class distribution")
plt.xlabel("Label (0=Real, 1=Fake)")
plt.ylabel("Count")
plt.show()

# save the plot
outpath = "plots/class_distribution.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()

#           distribution was ~45/55%

import matplotlib.pyplot as plt
import seaborn as sns

# title len (words)
df['title_len'] = df['title'].apply(lambda x: len(str(x).split()))

# histogram
plt.figure(figsize=(8,4))
sns.histplot(df['title_len'], bins=40)
plt.title("Title Length Distribution (word count)")
plt.xlabel("Number of words")
plt.ylabel("Frequency")
#plt.show()
# save the plot
outpath = "plots/title_length_hist.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()

# check outliers
df['title_len'].describe()
df['title_len'].quantile([0.9, 0.95, 0.99, 0.999])
df[df['title_len'] == 1]['title'].unique()[:50]


# remove NaN titles
df = df.dropna(subset=['title']).reset_index(drop=True)

# remove when titles have urls
before = len(df)
df = df[~df['title'].str.contains(settings.URL_PATTERN, case=False, regex=True)].reset_index(drop=True)
after = len(df)
print(f"Before: {before}, After: {after}, Removed: {before - after}")

# to do modulu preprocessing
df = df[df['title_len'] <= settings.MAX_TITLE_LEN].reset_index(drop=True) 
df = df[df['title_len'] >= settings.MIN_TITLE_LEN].reset_index(drop=True) 











df.to_csv('combined_fake_news_dataset.csv', index=False)
#eda
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df['title'].nunique(), df['text'].nunique())
#klasy sa zrownowazone

# check lenghts
df['text_length'] = df['text'].apply(len)
df['title_length'] = df['title'].apply(len)
df['text_words'] = df['text'].apply(lambda x: len(x.split()))
df['title_words'] = df['title'].apply(lambda x: len(x.split()))
#df['avg_word_len'] = df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]))
print(df.groupby('label')[['text_length','title_length']].describe())
print(df.groupby('label')[['text_words','title_words']].describe())
# check most common words by class
from collections import Counter
import re
from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def most_common_words(text_series, n=20):
    words = ' '.join(text_series).lower()
    words = re.findall(r'\b\w+\b', words)
    words = [w for w in words if w not in stop_words]
    return Counter(words).most_common(n)

# print("Fake news top words:", most_common_words(df[df['label']==1]['text']))
# print("True news top words:", most_common_words(df[df['label']==0]['text']))

top_fake = most_common_words(df[df['label']==1]['text'])
top_real = most_common_words(df[df['label']==0]['text'])

plt.bar(*zip(*top_fake))
plt.title("Top words in Fake News")
plt.show()

#workcloud visualisation

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text_fake = ' '.join(df[df['label']==1]['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_fake)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Fake news WordCloud")
plt.show()

# emotion/sentiment
from textblob import TextBlob

df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

print(f"sentiment: {df.groupby('label')['sentiment'].mean()}")
print(f"subjectivity: {df.groupby('label')['subjectivity'].mean()}")

#group by topic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

vectorizer = CountVectorizer(
    stop_words='english',
    max_df=0.8,
    min_df=10,
    max_features=5000
)

X = vectorizer.fit_transform(df['text'])

lda = LatentDirichletAllocation(
    n_components=20, # topics -k
    max_iter=15, 
    learning_method='batch',
    doc_topic_prior=0.15,   # α
    topic_word_prior=0.01, # β
    random_state=42,
    n_jobs=-1)
lda_features = lda.fit_transform(X)

df['topic'] = np.argmax(lda_features, axis=1)   




#end of eda

#1 TF-IDF and Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    max_df=0.9,   #ignore very common words
    min_df=9    #ignore very rare words
    )
X = vectorizer.fit_transform(df['text'])
y = df['label']

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

#1.1 TF-IDF and Logistic Regression with number of words as features
from scipy.sparse import hstack
X_extra = df[['text_words','title_words','text_length','title_length','avg_word_len']].values