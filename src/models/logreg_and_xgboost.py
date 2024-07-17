import warnings
from typing import Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

TOPICS = ["культура отмены", "феминизм", "ЛГБТК+", "эйджизм", "лукизм"]

BOW_VECTORIZER_PATH = "models/bow/bow_vectorizer.pkl"
TFIDF_VECTORIZER_PATH = "models/tfidf/tfidf_vectorizer.pkl"

LOGREG_PARAMS = {"penalty": ["l1", "l2"], "C": [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]}
XGBOOST_PARAMS = {
    "eta": [0.05, 0.2, 0.5],
    "gamma": [0, 0.1, 0.5, 1],
    "max_depth": [3, 5, 7],
}


def balancing_data(train_data: pd.DataFrame) -> pd.DataFrame:
    """
    Balancing data by labels

    Params:
    -------
        train_data (pd.DataFrame): train dataframe

    Returns:
    --------
        (pd.DataFrame): new train dataframe with balanced labels
    """
    labels_train = []
    for i in range(3):
        labels_train.append(train_data[train_data.stance == i])

    minority = min([labels_train[i].shape[0] for i in range(3)])

    labels_train_min = []
    for i in range(3):
        labels_train_min.append(labels_train[i].sample(minority))

    train_data = pd.concat(labels_train_min, ignore_index=True)

    return train_data.sample(frac=1).reset_index(drop=True)


def make_train_valid(
    data: pd.DataFrame,
    topics: List[str] = TOPICS,
    test_shape: int = 8,
    balancing: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Making train and valid dataframes

    Params:
    -------
        data (pd.DataFrame): input data
        topics (List[str]): list of topics at dataframe
        test_shape (int): ratio train and valid dataframes' shape
        balancing (str): type of balancing data

    Returns:
    --------
        train_data, valid_data (Tuple[pd.DataFrame, pd.DataFrame]): train and valid dataframes
    """
    sample = data.sample(frac=1)
    train_data = sample.iloc[data.shape[0] // 100 * test_shape :]
    valid_data = sample.iloc[: data.shape[0] // 100 * test_shape]

    if balancing == "all":
        train_data = balancing_data(train_data)
    elif balancing == "each":
        all_topics = []
        for topic in topics:
            train_data_topic = train_data[train_data.topic == topic]
            train_data_topic = balancing_data(train_data_topic)
            all_topics.append(train_data_topic)

        train_data = pd.concat(all_topics, ignore_index=True)

        train_data = train_data.sample(frac=1).reset_index(drop=True)

    return train_data, valid_data


def bow_for_train(
    X_train_content: pd.DataFrame,
    save: bool = True,
    save_path: str = BOW_VECTORIZER_PATH,
) -> Any:
    """
    Making bag of words vectorizer for train dataframe

    Params:
    -------
        X_train_content (pd.DataFrame): content from train dataframe
        save (bool): flag for saving
        save_path (str): path to saving vectorizer

    Returns:
    --------
        bow_vectorizer (Any): trained vectorizer
    """
    bow_vectorizer = CountVectorizer()
    bow_vectorizer.fit(X_train_content)

    if save:
        joblib.dump(bow_vectorizer, save_path)

    return bow_vectorizer


def tfidf_for_train(
    X_train_content: pd.DataFrame,
    save: bool = True,
    save_path: str = TFIDF_VECTORIZER_PATH,
) -> Any:
    """
    Making tfidf vectorizer for train dataframe

    Params:
    -------
        X_train_content (pd.DataFrame): content from train dataframe
        save (bool): flag for saving
        save_path (str): path to saving vectorizer

    Returns:
    --------
        bow_vectorizer (Any): trained vectorizer
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X_train_content)

    if save:
        joblib.dump(tfidf_vectorizer, save_path)

    return tfidf_vectorizer


def vectorize_train_valid(
    data: pd.DataFrame,
    vectorizer_type: str,
    balancing: str = None,
    topic_feature: bool = True,
    save: bool = True,
) -> Tuple[Any, Any, Any, Any]:
    """
    Making vectors for train and valid data

    Params:
    -------
        data (pd.DataFrame): input data
        vectorizer_type (str): type of vectorizer
        balancing (str): type of balancing
        topic_feature (bool): flag for including topic as a feature
        save (bool): flag for saving

    Returns:
    --------
        X_train_bow, X_valid_bow, y_train, y_valid (Tuple[Any, Any, Any, Any]): train and valid tfidf-vectors and labels
    """
    X_train, X_valid = make_train_valid(data, balancing=balancing)
    y_train = X_train.stance.values
    y_valid = X_valid.stance.values

    if topic_feature:
        if vectorizer_type == "bow":
            vectorizer = bow_for_train(X_train.content.values, save)
        elif vectorizer_type == "tfidf":
            vectorizer = tfidf_for_train(X_train.content.values, save)
        else:
            print("Choose bow or tfidf vectorizer")
            return

        X_train_content_vect = vectorizer.transform(X_train.content.values)
        X_valid_content_vect = vectorizer.transform(X_valid.content.values)

        X_train_topic_sm = sp.csc_matrix(X_train.topic.values)
        X_valid_topic_sm = sp.csc_matrix(X_valid.topic.values)

        X_train_vect = sp.hstack(
            [
                X_train_content_vect,
                np.reshape(X_train_topic_sm, (X_train_topic_sm.shape[1], 1)),
            ],
            format="csr",
        )
        X_valid_vect = sp.hstack(
            [
                X_valid_content_vect,
                np.reshape(X_valid_topic_sm, (X_valid_topic_sm.shape[1], 1)),
            ],
            format="csr",
        )
    else:
        if vectorizer_type == "bow":
            vectorizer = bow_for_train(X_train.content.values, save)
        elif vectorizer_type == "tfidf":
            vectorizer = tfidf_for_train(X_train.content.values, save)
        else:
            print("Choose bow or tfidf vectorizer")
            return

        X_train_vect = vectorizer.transform(X_train.content.values)
        X_valid_vect = vectorizer.transform(X_valid.content.values)

    return X_train_vect, X_valid_vect, y_train, y_valid


def best_model(
    data: pd.DataFrame,
    model_name: str,
    vectorizer_type: str,
    balancing: str = None,
    topic_feature: bool = True,
    save: bool = True,
) -> None:
    """
    Getting best trained model for input data

    Params:
    -------
        data (pd.DataFrame): input data
        model_name (str): name of trained model
        vectorizer_type (str): type of vectorizer
        balancing (str): type of balancing
        topic_feature (bool): flag for including topic as a feature
        save (bool): flag for saving data after vectorizing
    """
    X_train, X_valid, y_train, y_valid = vectorize_train_valid(
        data, vectorizer_type, balancing, topic_feature, save
    )

    if model_name == "logreg":
        model_params = LOGREG_PARAMS
        model = LogisticRegression()
    elif model_name == "xgboost":
        model_params = XGBOOST_PARAMS
        model = XGBClassifier()
    else:
        print("Choose LogReg or XGBoost model")
        return

    grid = GridSearchCV(model, model_params, cv=3, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    best_model = grid.best_estimator_

    print("best parameters for model:\t", model_name)
    print(best_params)
    print()

    joblib.dump(best_model, f"models/best_model_{model_name}_{vectorizer_type}.pkl")

    preds = best_model.predict(X_valid)

    print(classification_report(y_valid, preds))
