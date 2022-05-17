import re
import nltk
import string
from nltk.corpus import stopwords
from pymystem3 import Mystem

def lowercasing(tweet):
    return tweet.lower()

def delete_punctuation(tweet):
    return re.sub(r'[^\w\s]','', tweet) 

def lemmatization(tweet):
    mystem_analyzer = Mystem()
    lemmas = mystem_analyzer.lemmatize(tweet)
    new_tweet = ' '.join(lemmas)

    return new_tweet

def delete_stopwords(tweet):
    russian_stopwords = stopwords.words("russian")
    split_tweet = tweet.split()
    new_split_tweet = [word for word in split_tweet if word not in russian_stopwords]
    new_tweet = ' '.join(new_split_tweet)

    return new_tweet

def preprocess(data):
    data['content'] = data.content.apply(lowercasing)
    data['content'] = data.content.apply(delete_punctuation)
    data['content'] = data.content.apply(lemmatization)
    data['content'] = data.content.apply(delete_stopwords)
    
    return data
