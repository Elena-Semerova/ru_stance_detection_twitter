import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="darkgrid")

def make_date_features(data):
    data['date'] = pd.to_datetime(data['date'])
    data['day'] = data['date'].dt.strftime('%d')
    data['month'] = data['date'].dt.strftime('%m')
    data['year'] = data['date'].dt.strftime('%y')
    data['year_month'] = data['date'].dt.strftime('%y-%m')

    return data

def visualize_year_month(data, topic_name=''):
    data_date = make_date_features(data)
    df_groupby = data_date.groupby(['year_month']).count()['content']
    df_count = pd.DataFrame({'year_month' : data_date.year_month.unique(), 'count' : df_groupby.values})

    sns.relplot(data=df_count, x='year_month', y='count', kind='line', height=5, aspect=3)

    plt.title('Dependence of number of tweets on the year_month for topic: ' + topic_name, size = 15)
    plt.xlabel('year_month', fontsize = 12)
    plt.ylabel('number of tweets', fontsize = 12)

    plt.show()
