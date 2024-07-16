import re
import pandas as pd
from nltk.corpus import stopwords
from pymystem3 import Mystem

def lowercasing(tweet: str) -> str:
    return tweet.lower()

def delete_punctuation(tweet: str) -> str:
    return re.sub(r'[^\w\s]','', tweet) 

def lemmatization(tweet: str) -> str:
    mystem_analyzer = Mystem()
    lemmas = mystem_analyzer.lemmatize(tweet)
    new_tweet = ' '.join(lemmas)

    return new_tweet

def delete_stopwords(tweet: str) -> str:
    russian_stopwords = stopwords.words("russian")
    split_tweet = tweet.split()
    new_split_tweet = [word for word in split_tweet if word not in russian_stopwords]
    new_tweet = ' '.join(new_split_tweet)

    return new_tweet

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data['content'] = data.content.apply(lowercasing)
    data['content'] = data.content.apply(delete_punctuation)
    data['content'] = data.content.apply(lemmatization)
    data['content'] = data.content.apply(delete_stopwords)
    
    return data
