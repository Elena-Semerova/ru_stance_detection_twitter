import pandas as pd
import snscrape.modules.twitter as sntwitter
import itertools
from tqdm import tqdm

def find_tweets(search_string: str, date: str = '', num_tweets: int = 10000000000) -> pd.DataFrame:
    scraped_tweets = sntwitter.TwitterSearchScraper(search_string + date).get_items()
    sliced_scraped_tweets = itertools.islice(scraped_tweets, num_tweets)
    tweets = pd.DataFrame(sliced_scraped_tweets)
    
    return tweets

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data_clean = data[data.lang == 'ru']
    data_clean = data_clean[['url', 'date', 'content', 'id']]
    
    return data_clean

def make_dataframe(keywords: list, date: str = '') -> pd.DataFrame:
    data = find_tweets(keywords[0], date)

    for i in tqdm(range(1, len(keywords))):
        keyword = keywords[i]
        data_i = find_tweets(keyword, date)
        data = pd.concat([data, data_i], ignore_index=True)

    return clean_data(data)

def make_examples(data: pd.DataFrame, size: int = 5) -> pd.DataFrame:
    for tweet in data.sample(size).content.values:
        print(tweet)
        print()
