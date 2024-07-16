import re

import pandas as pd
from nltk.corpus import stopwords
from pymystem3 import Mystem

LANGUAGE = "russian"


def lowercasing(tweet: str) -> str:
    """
    Lowercasing symbols at tweet

    Params:
    -------
        tweet (str): input tweet

    Returns:
    --------
        (str): new tweet
    """
    return tweet.lower()


def delete_punctuation(tweet: str) -> str:
    """
    Deleting punctuation at tweet

    Params:
    -------
        tweet (str): input tweet

    Returns:
    --------
        (str): tweet without punctuation
    """
    return re.sub(r"[^\w\s]", "", tweet)


def lemmatization(tweet: str) -> str:
    """
    Lematizating words at tweet

    Params:
    -------
        tweet (str): input tweet

    Returns:
    --------
        new_tweet (str): tweet with lemmatizated words
    """
    mystem_analyzer = Mystem()
    lemmas = mystem_analyzer.lemmatize(tweet)
    new_tweet = " ".join(lemmas)

    return new_tweet


def delete_stopwords(tweet: str, language: str = LANGUAGE) -> str:
    """
    Deleting stopwords from tweet

    Params:
    -------
        tweet (str): input tweet
        language (str): language of stopwords

    Returns:
    --------
        new_tweet (str): tweet without stopwords
    """
    stopwords_lang = stopwords.words(language)
    split_tweet = tweet.split()
    new_split_tweet = [word for word in split_tweet if word not in stopwords_lang]
    new_tweet = " ".join(new_split_tweet)

    return new_tweet


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing dataframe with tweets:
        - lowercasing
        - deleting punctuation
        - lemmatizating
        - deleting stopwords

    Params:
    -------
        data (pd.DataFrame): input dataframe

    Returns:
    --------
        data (pd.DataFrame): preprocessed dataframe
    """
    data["content"] = data.content.apply(lowercasing)
    data["content"] = data.content.apply(delete_punctuation)
    data["content"] = data.content.apply(lemmatization)
    data["content"] = data.content.apply(delete_stopwords)

    return data
