from typing import List

import pandas as pd


def read_data(topic_name: str, path: str) -> pd.DataFrame:
    """
    Reading data and filtering by topic

    Params:
    -------
        topic_name (str): topic
        path (str): path with data

    Returns:
    --------
        (pd.DataFrame): final dataframe
    """
    data = pd.read_csv(path)
    data["topic"] = topic_name

    return data[["content", "topic"]]


def make_dataset(
    topics: List[str], topics_ru: List[str], shuffle: bool = False, save: bool = False
) -> pd.DataFrame:
    """
    Making total dataset with all topics

    Params:
    -------
        topics (List[str]): list of topics
        topics_ru (List[str]): list of topics on Russian
        shuffle (bool): flag for shuffling
        save (bool): flag for saving total dataset

    Returns:
    --------
        data (pd.DataFrame): total dataset
    """
    # to do: name config
    data = pd.DataFrame({"content": [], "topic": []})
    for topic in topics:
        # clean_data
        path = topic.upper() + "_PATH"
        topic_name = topics_ru[topic]
        data_i = read_data(topic_name, globals()[path])
        data = pd.concat([data, data_i], ignore_index=True)

    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)

    if save:
        data.to_csv("data/data_for_labeling.csv", index=False)

    return data
