import numpy as np
import pandas as pd

from typing import Dict, List

LABELS_DICT = {
    'stance': ['favor', 'against', 'neutral', 'error'],
    'sentiment': ['positive', 'negative', 'neutral']
}


def aggregate_target(data: pd.DataFrame, target_feature: str, labels_dict: Dict[str, List[str]] = LABELS_DICT) -> pd.DataFrame:
    """
    Aggregating labels after toloka

    Params:
    -------
        data (pd.DataFrame): dataframe with answers after toloka
        target_feature (str): target to aggregate
        labels_dict (Dict[str, List[str]]): dict with names of targets and possible labels

    Returns:
    --------
        (pd.DataFrame): dataframe with final labels
    """
    data = data.sort_values(by=["content", "skill"])
    data = data.set_index(pd.Series([i for i in range(data.shape[0])]))
    data["true"] = [""] * data.shape[0]

    for i in range(0, data.shape[0], 3):
        labels = [data.iloc[j].stance for j in range(i, i + 3)]
        weights = [data.iloc[j].weight for j in range(i, i + 3)]
        probs = [[0]] * len(LABELS_DICT[target_feature])

        for j in range(3):
            label_index = LABELS_DICT[target_feature].index(labels[j])
            probs[label_index] += weights[j] / 3

        true_idx = np.argmax(probs)

        for j in range(i, i + 3):
            data.loc[j, "true"] = LABELS_DICT[target_feature][true_idx]

    data = data[data[target_feature] == data.true]
    data = data.drop_duplicates(subset=["content"], keep="first")

    return data[["topic", "content", target_feature]]