import itertools
from typing import List

import pandas as pd
import snscrape.modules.twitter as sntwitter
from tqdm import tqdm

MAX_NUM_TWEETS = 10000000000
LANGUAGE = "ru"
COUNT_OF_SAMPLES = 5


def find_tweets(
    search_string: str, date: str = "", num_tweets: int = MAX_NUM_TWEETS
) -> pd.DataFrame:
    """
    Finding tweets that include search_string for period

    Params:
    -------
        search_string (str): string that should be included
        date (str): period for searching
        num_tweets (int): limit for count of tweets

    Returns:
    --------
        tweets (pd.DataFrame): dataframe with finded tweets
    """
    scraped_tweets = sntwitter.TwitterSearchScraper(search_string + date).get_items()
    sliced_scraped_tweets = itertools.islice(scraped_tweets, num_tweets)
    tweets = pd.DataFrame(sliced_scraped_tweets)

    return tweets


def clean_data(data: pd.DataFrame, language: str = LANGUAGE) -> pd.DataFrame:
    """
    Cleaning dataframe with tweets by language and choosing columns

    Params:
    -------
        data (pd.DataFrame): imput dataframe
        language (str): language for filtering

    Returns:
    --------
        data_clean (pd.DataFrame): cleaned dataframe
    """
    data_clean = data[data.lang == language]
    data_clean = data_clean[["url", "date", "content", "id"]]

    return data_clean


def make_dataframe(
    keywords: List[str], date: str = "", language: str = LANGUAGE
) -> pd.DataFrame:
    """
    Making dataframe with tweets that include keywords from list for period

    Params:
    -------
        keywords (List[str]): list with keywords
        date (str): period for searching
        language (str): language for filtering

    Returns:
    --------
        cleaned_data (pd.DataFrame): final dataframe
    """
    data = find_tweets(keywords[0], date)

    for i in tqdm(range(1, len(keywords))):
        keyword = keywords[i]
        data_i = find_tweets(keyword, date)
        data = pd.concat([data, data_i], ignore_index=True)

    cleaned_data = clean_data(data, language)

    return cleaned_data


def make_examples(data: pd.DataFrame, size: int = COUNT_OF_SAMPLES) -> None:
    """
    Printing some examples from input dataframe

    Params:
    -------
        data (pd.DataFrame): input dataframe
        size (int): count of examples
    """
    for tweet in data.sample(size).content.values:
        print(tweet)
        print()
