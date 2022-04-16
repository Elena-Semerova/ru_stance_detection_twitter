import pandas as pd
import re
import warnings

warnings.filterwarnings('ignore')

def replacing(symbols, replacement, tweet):
    symbols = [element for element in symbols if element in tweet]
    
    for symbol in symbols:
        tweet = tweet.replace(symbol, replacement)
        
    return tweet

def cleaning_tweet(tweet):
    tweet = replacing(
        '\u00AB\u00BB\u2039\u203A\u201E\u201A\u201C\u201F\u2018\u201B\u201D\u2019',
        '\u0022',
        tweet
    )

    tweet = replacing(
        '\u2012\u2013\u2014\u2015\u203E\u0305\u00AF',
        '\u2003\u002D\u002D\u2003',
        tweet
    )
    
    tweet = replacing(
        '\u2000\u2001\u2002\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B\u202F\u205F\u2060\u3000',
        '\u2002',
        tweet
    )

    tweet = replacing(
        '\u02CC\u0307\u0323\u2022\u2023\u2043\u204C\u204D\u2219\u25E6\u00B7\u00D7\u22C5\u2219\u2062',
        '.',
         tweet
    )

    tweet = replacing('\u2010\u2011', '\u002D', tweet)
    tweet = replacing('\u2217', '\u002A', tweet)
    tweet = replacing('…', '...', tweet)
    tweet = replacing('\u00C4', 'A', tweet)
    tweet = replacing('\u00E4', 'a', tweet)
    tweet = replacing('\u00CB', 'E', tweet)
    tweet = replacing('\u00EB', 'e', tweet)
    tweet = replacing('\u1E26', 'H', tweet)
    tweet = replacing('\u1E27', 'h', tweet)
    tweet = replacing('\u00CF', 'I', tweet)
    tweet = replacing('\u00EF', 'i', tweet)
    tweet = replacing('\u00D6', 'O', tweet)
    tweet = replacing('\u00F6', 'o', tweet)
    tweet = replacing('\u00DC', 'U', tweet)
    tweet = replacing('\u00FC', 'u', tweet)
    tweet = replacing('\u0178', 'Y', tweet)
    tweet = replacing('\u00FF', 'y', tweet)
    tweet = replacing('\u00DF', 's', tweet)
    tweet = replacing('\u1E9E', 'S', tweet)
    tweet = replacing('\t\r\n', ' ', tweet)
    
    tweet = re.sub('&gt;', '>', tweet)
    tweet = re.sub('&lt;', '<', tweet)
    tweet = re.sub('&le;', '<=', tweet)
    tweet = re.sub('&ge;', '>=', tweet)
    tweet = re.sub('\u2003\u2003', '\u2003', tweet)
    tweet = re.sub('\t\t', '\t', tweet)
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\S+ ", "", tweet)

    currencies = list(
        '\u0022\u20BD\u0024\u00A3\u20A4\u20AC\u20AA\u2133\u20BE\u00A2\u058F\u0BF9\u20BC\u20A1\u20A0\u20B4\u20A7\u20B0\u20BF\u20A3\u060B\u0E3F\u20A9\u20B4\u20B2\u0192\u20AB\u00A5\u20AD\u20A1\u20BA\u20A6\u20B1\uFDFC\u17DB\u20B9\u20A8\u20B5\u09F3\u20B8\u20AE\u0192'
    )

    alphabet = list(
        'абвгдеёзжийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЗЖИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
    )
    
    numbers = list(
        '0123456789'
    )
    
    punctuation = list(
        ',.[]{}()=+-−*&^%$#@!~;:§/\|\?\'<>'
    )

    allowed = set(currencies + alphabet + numbers + punctuation)

    cleaned_tweet = [symbol for symbol in tweet if symbol in allowed]
    cleaned_tweet = ''.join(cleaned_tweet)

    return cleaned_tweet

def cleaning_data(data):
    data = data[['url', 'date', 'content', 'id']]
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')
    
    data_tweets = data.content.values
    cleaned_data_tweets = []
    
    for tweet in data_tweets:
        split_cleaned_tweet = cleaning_tweet(tweet)
        cleaned_data_tweets.append(' '.join(split_cleaned_tweet))
        
    data['content'] = cleaned_data_tweets
    data = data.drop_duplicates(subset=['content'], keep='first')

    return data

def saving(data, topic_name):
    path = 'data/' + topic_name + '_clean.csv'
    data.to_csv(path)

def preprocess(data, topic_name, save=False):
    data = cleaning_data(data)
    print('Cleaning is done')
    
    if save:
        saving(data, topic_name)
        print('Saving is done')
        
    return data
