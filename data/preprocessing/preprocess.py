import pandas as pd

def cleaning(data):
    data = data[['url', 'date', 'content', 'id']]
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')
    data = data.drop_duplicates(subset=['content'], keep='first')

    return data

def saving(data, topic_name):
    path = 'data/' + topic_name + '_clean.csv'
    data.to_csv(path)

def preprocess(data, topic_name, save=False):
    data = cleaning(data)
    print('Cleaning is done')
    
    if save:
        saving(data, topic_name)
        print('Saving is done')
