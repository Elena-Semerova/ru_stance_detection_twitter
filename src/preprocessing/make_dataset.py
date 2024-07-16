import pandas as pd
from ru_stance_detection_twitter.configs.label_config import *

def read_data(topic_name, path):
    data = pd.read_csv(path)
    data['topic'] = topic_name
    
    return data[['content', 'topic']]

def make_dataset(topics, topics_ru, shuffle=False, save=False):
    data = pd.DataFrame({'content':[], 'topic':[]})
    for topic in topics:
        # clean_data
        path = topic.upper() + '_PATH'
        topic_name = topics_ru[topic]
        data_i = read_data(topic_name, globals()[path])
        data = pd.concat([data, data_i], ignore_index=True)
        
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    
    if save:
        data.to_csv('data/data_for_labeling.csv', index=False)
    
    return data
