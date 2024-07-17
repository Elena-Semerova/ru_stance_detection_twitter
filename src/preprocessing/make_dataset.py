from typing import List, Dict

import pandas as pd

from src.utils import read_yml_file

CONFIG_PATH = "configs/name_config.yaml"


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
    topics: List[str],
    topics_ru: Dict[str, str],
    shuffle: bool = False,
    save: bool = False,
    config_path: str = CONFIG_PATH,
) -> pd.DataFrame:
    """
    Making total dataset with all topics

    Params:
    -------
        topics (List[str]): list of topics
        topics_ru (List[str]): list of topics on Russian
        shuffle (bool): flag for shuffling
        save (bool): flag for saving total dataset
        config_path (str): path to config file

    Returns:
    --------
        data (pd.DataFrame): total dataset
    """
    name_config = read_yml_file(config_path)
    data = pd.DataFrame({"content": [], "topic": []})

    for topic in topics:
        path = "clean_data_" + topic + "_path"
        topic_name = topics_ru[topic]
        data_i = read_data(topic_name, name_config[path])
        data = pd.concat([data, data_i], ignore_index=True)

    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)

    if save:
        data.to_csv(name_config["unlabeled_all_data_path"], index=False)

    return data
