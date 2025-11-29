import settings
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud, STOPWORDS


def run_eda():
    # 1.Load datasets
    df_1 = pd.DataFrame(utils.download_dataset(settings.KAGGLE_DATASET_NAME))
    df_2 = pd.DataFrame(utils.download_dataset(settings.KAGGLE_DATASET_NAME_2))
    df = pd.concat([df_1, df_2], ignore_index=True)

    print(df.head())
    print(df.info())
    print(df['label'].value_counts())

    # 2. Remove NaN titles
    before = len(df)
    df = df.dropna(subset=['title']).reset_index(drop=True)
    print(f"Removed NaN titles. Before: {before}, After: {len(df)}, Removed: {before - len(df)}")


    # 3. Find duplicates by title
    before = len(df)
    dupes = df[df.duplicated(subset=["title"], keep=False)]

    # 3.1 duplicate titles with both labels - remove all of them
    conflicts = dupes.groupby("title")["label"].nunique()
    conflict_titles = conflicts[conflicts == 2].index.tolist()
    df = df[~df["title"].isin(conflict_titles)]

    # 3.2 remove normal duplicated - keep first
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    # 
    print(rf"Removed duplicates. Size before: {before}, after: {len(df)}, removed: {before - len(df)}")

    # 4. titles cleanup
    df['title_len'] = df['title'].apply(lambda x: len(str(x).split()))
    # see the distribution of title lengths
    plt.figure(figsize=(8,4))
    sns.histplot(df['title_len'], bins=40)
    plt.title("Title Length Distribution (word count)")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    #plt.show()
    # save the plot
    outpath = "plots/title_length_hist.png"
    # make sure plots directory exists
    os.makedirs("plots", exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    # 4.1 check and remove outliers
    df['title_len'].describe()
    df['title_len'].quantile([0.9, 0.95, 0.99, 0.999])
    df[df['title_len'] == 1]['title'].unique()[:50]

    df = df[df['title_len'] <= settings.MAX_TITLE_LEN].reset_index(drop=True) 
    df = df[df['title_len'] >= settings.MIN_TITLE_LEN].reset_index(drop=True) 

    # 4.2 Remove when titles have urls
    before = len(df)
    df = df[~df['title'].str.contains(settings.URL_PATTERN, case=False, regex=True)].reset_index(drop=True)
    print(f"Remove titles with URL. Before: {before}, After: {len(df)}, Removed: {before - len(df)}")


    # 5. see the class distribution of labels
    plt.figure(figsize=(6,4))
    sns.countplot(x=df['label'])
    plt.title("Class distribution")
    plt.xlabel("Label (0=Real, 1=Fake)")
    plt.ylabel("Count")
    #plt.show()
    # save the plot
    outpath = "plots/class_distribution.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

    print("Class distribution (percent):")
    print(df['label'].value_counts(normalize=True) * 100)



    # 6. see the distribution of text lengths
    df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(8,4))
    sns.histplot(df['text_len'], bins=30)
    plt.title("Text Length Distribution (word count)")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.savefig("plots/text_length_hist.png", dpi=150, bbox_inches="tight")
    plt.close()


    # 7. most common words in titles for fake vs real
    #download stopwords if not present
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))
    # add additional stopwords from settings
    stop_words |= settings.ADDITIONAL_STOPWORDS

    def most_common_words(text_series, n=settings.NO_OF_WORDS):
        tokens = []
        for text in text_series:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
            words = [w for w in words if w not in stop_words]
            tokens.extend(words)
        return Counter(tokens).most_common(n)
    # fake
    print("Fake news top title words:\n", most_common_words(df[df['label']==1]['title']))
    # real
    print("\nReal news top title words:\n", most_common_words(df[df['label']==0]['title']))

    # 8. Wordclouds for fake vs real titles

    fake_top = most_common_words(df[df['label']==1]['title'], n=settings.WORD_CLOUD_TOP_N)
    real_top = most_common_words(df[df['label']==0]['title'], n=settings.WORD_CLOUD_TOP_N)

    fake_dict = dict(fake_top)
    real_dict = dict(real_top)

    # fake WordCloud
    wc_fake = WordCloud(
        width=1600,
        height=900,
        background_color="white"
    ).generate_from_frequencies(fake_dict)
    wc_fake.to_file("plots/wordcloud_fake.png")

    # real WordCloud
    wc_real = WordCloud(
        width=1600,
        height=900,
        background_color="white"
    ).generate_from_frequencies(real_dict)
    wc_real.to_file("plots/wordcloud_real.png")

    # 9. Save cleaned dataset
    try:
        df.to_csv("data/clean_dataset.csv", index=False)
        print("Cleaned dataset saved to data/clean_dataset.csv")
    except Exception as e:
        print(f"Error saving cleaned dataset: {e}")