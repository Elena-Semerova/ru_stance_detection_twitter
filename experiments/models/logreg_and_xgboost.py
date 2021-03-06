import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

def balancing_data(train_data):
    train_0 = train_data[train_data.stance == 0]
    train_1 = train_data[train_data.stance == 1]
    train_2 = train_data[train_data.stance == 2]

    minority = min([train_0.shape[0], train_1.shape[0], train_2.shape[0]])

    train_0 = train_0.sample(minority)
    train_1 = train_1.sample(minority)
    train_2 = train_2.sample(minority)

    train_data = pd.concat([train_0, train_1, train_2], ignore_index=True)

    return train_data.sample(frac=1).reset_index(drop=True)

def make_train_valid(data, test_shape=8, balancing=None):
    sample = data.sample(frac=1)
    train_data = sample.iloc[data.shape[0] // 100 * test_shape:]
    valid_data = sample.iloc[:data.shape[0] // 100 * test_shape]

    if balancing == 'all':
        train_data = balancing_data(train_data)
    elif balancing == 'each':
        train_data_cc = train_data[train_data.topic == 'культура отмены']
        train_data_fem = train_data[train_data.topic == 'феминизм']
        train_data_lgbt = train_data[train_data.topic == 'ЛГБТК+']
        train_data_age = train_data[train_data.topic == 'эйджизм']
        train_data_look = train_data[train_data.topic == 'лукизм']

        train_data_cc = balancing_data(train_data_cc)
        train_data_fem = balancing_data(train_data_fem)
        train_data_lgbt = balancing_data(train_data_lgbt)
        train_data_age = balancing_data(train_data_age)
        train_data_look = balancing_data(train_data_look)

        train_data = pd.concat([train_data_cc, train_data_fem, train_data_lgbt,
                                train_data_age, train_data_look], ignore_index=True)
        
        train_data = train_data.sample(frac=1).reset_index(drop=True)
    
    return train_data, valid_data

def bow_for_train(X_train_content, save=True):
    bow_vectorizer = CountVectorizer()
    X_train_bow = bow_vectorizer.fit_transform(X_train_content)
    
    if save:
        joblib.dump(bow_vectorizer, 'bow/bow_vectorizer' + '.pkl')
        
    return bow_vectorizer

def tfidf_for_train(X_train_content, save=True):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_bow = tfidf_vectorizer.fit_transform(X_train_content)
    
    if save:
        joblib.dump(tfidf_vectorizer, 'tfidf/tfidf_vectorizer' + '.pkl')
        
    return tfidf_vectorizer

def bow_train_valid(data, balancing=None, topic_feature=True, save=True):
    X_train, X_valid = make_train_valid(data)
    
    if topic_feature:
        X_train_content = X_train.content.values
        X_train_topic = X_train.topic.values
        y_train = X_train.stance.values
        X_valid_content = X_valid.content.values
        X_valid_topic = X_valid.topic.values
        y_valid = X_valid.stance.values

        bow_vectorizer = bow_for_train(X_train_content, save)

        X_train_content_bow = bow_vectorizer.transform(X_train_content)
        X_valid_content_bow = bow_vectorizer.transform(X_valid_content)

        X_train_topic_sm = sp.csc_matrix(X_train_topic)
        X_valid_topic_sm = sp.csc_matrix(X_valid_topic)

        X_train_bow = sp.hstack([X_train_content_bow, np.reshape(X_train_topic_sm, (X_train_topic_sm.shape[1], 1))], format='csr')
        X_valid_bow = sp.hstack([X_valid_content_bow, np.reshape(X_valid_topic_sm, (X_valid_topic_sm.shape[1], 1))], format='csr')
    else:
        X_train_content = X_train.content.values
        y_train = X_train.stance.values
        
        X_valid_content = X_valid.content.values
        y_valid = X_valid.stance.values
        
        bow_vectorizer = bow_for_train(X_train_content, save)
        
        X_train_bow = bow_vectorizer.transform(X_train_content)
        X_valid_bow = bow_vectorizer.transform(X_valid_content)
    
    return X_train_bow, X_valid_bow, y_train, y_valid

def tfidf_train_valid(data, balancing=None, topic_feature=True, save=True):
    X_train, X_valid = make_train_valid(data)
    
    if topic_feature:
        X_train_content = X_train.content.values
        X_train_topic = X_train.topic.values
        y_train = X_train.stance.values
        X_valid_content = X_valid.content.values
        X_valid_topic = X_valid.topic.values
        y_valid = X_valid.stance.values

        tfidf_vectorizer = tfidf_for_train(X_train_content, save)

        X_train_content_tfidf = tfidf_vectorizer.transform(X_train_content)
        X_valid_content_tfidf = tfidf_vectorizer.transform(X_valid_content)

        X_train_topic_sm = sp.csc_matrix(X_train_topic)
        X_valid_topic_sm = sp.csc_matrix(X_valid_topic)

        X_train_tfidf = sp.hstack([X_train_content_tfidf, np.reshape(X_train_topic_sm, (X_train_topic_sm.shape[1], 1))], format='csr')
        X_valid_tfidf = sp.hstack([X_valid_content_tfidf, np.reshape(X_valid_topic_sm, (X_valid_topic_sm.shape[1], 1))], format='csr')
    else:
        X_train_content = X_train.content.values
        y_train = X_train.stance.values
        
        X_valid_content = X_valid.content.values
        y_valid = X_valid.stance.values
        
        tfidf_vectorizer = tfidf_for_train(X_train_content, save)
        
        X_train_tfidf = tfidf_vectorizer.transform(X_train_content)
        X_valid_tfidf = tfidf_vectorizer.transform(X_valid_content)
    
    return X_train_tfidf, X_valid_tfidf, y_train, y_valid

def best_model(data, model_name, vec_name, balancing=None, topic_feature=True, save=True):    
    if vec_name == 'bow':
        X_train, X_valid, y_train, y_valid = bow_train_valid(data, balancing, topic_feature)
    elif vec_name == 'tfidf':
        X_train, X_valid, y_train, y_valid = tfidf_train_valid(data, balancing, topic_feature)
    else:
        print('Choose bow or tfidf vectorizer')
        return
    
    if model_name == 'logreg':
        model_params = {
            'penalty' : ['l1', 'l2'],
            'C' : [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]
        }
        model = LogisticRegression()
    elif model_name == 'xgboost':
        model_params = {
            'eta' : [0.05, 0.2, 0.5],
            'gamma': [0, 0.1, 0.5, 1],
            'max_depth': [3, 5, 7]
        }
        model = XGBClassifier()
    else:
        print('Choose LogReg or XGBoost model')
        return
        
    
    grid = GridSearchCV(model, model_params, cv=3, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_params = grid.best_params_
    best_model = grid.best_estimator_

    print('best parameters for model:\t', model_name)
    print(best_params)
    print()
    
    joblib.dump(best_model, 'models/best_model_' + model_name + '_bow.pkl')
    
    preds = best_model.predict(X_valid)

    print(classification_report(y_valid, preds))