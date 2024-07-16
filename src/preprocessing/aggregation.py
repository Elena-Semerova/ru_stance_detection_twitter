import numpy as np
import pandas as pd


def aggeregate_stance(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregating labels with stances after toloka

    Params:
    -------
        data (pd.DataFrame): dataframe with answers after toloka

    Returns:
    --------
        (pd.DataFrame): dataframe with final labels
    """
    data = data.sort_values(by=["content", "skill"])
    data = data.set_index(pd.Series([i for i in range(data.shape[0])]))
    data["true"] = [""] * data.shape[0]

    for i in range(0, data.shape[0], 3):
        stance_1 = data.iloc[i].stance
        stance_2 = data.iloc[i + 1].stance
        stance_3 = data.iloc[i + 2].stance

        weight_1 = data.iloc[i].weight
        weight_2 = data.iloc[i + 1].weight
        weight_3 = data.iloc[i + 2].weight

        stances = [stance_1, stance_2, stance_3]
        weights = [weight_1, weight_2, weight_3]
        # favor, against, neutral, error
        probs = [0, 0, 0, 0]

        for j in range(3):
            if stances[j] == "favor":
                probs[0] += weights[j] / 3
            if stances[j] == "against":
                probs[1] += weights[j] / 3
            if stances[j] == "neutral":
                probs[2] += weights[j] / 3
            if stances[j] == "error":
                probs[3] += weights[j] / 3

        true_idx = np.argmax(probs)
        if true_idx == 0:
            data.loc[i, "true"] = "favor"
            data.loc[i + 1, "true"] = "favor"
            data.loc[i + 2, "true"] = "favor"
        elif true_idx == 1:
            data.loc[i, "true"] = "against"
            data.loc[i + 1, "true"] = "against"
            data.loc[i + 2, "true"] = "against"
        elif true_idx == 2:
            data.loc[i, "true"] = "neutral"
            data.loc[i + 1, "true"] = "neutral"
            data.loc[i + 2, "true"] = "neutral"
        elif true_idx == 3:
            data.loc[i, "true"] = "error"
            data.loc[i + 1, "true"] = "error"
            data.loc[i + 2, "true"] = "error"

    data = data[data.stance == data.true]
    data = data.drop_duplicates(subset=["content"], keep="first")

    return data[["topic", "content", "stance"]]


def aggregate_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregating labels with sentiments after toloka

    Params:
    -------
        data (pd.DataFrame): dataframe with answers after toloka

    Returns:
    --------
        (pd.DataFrame): dataframe with final labels
    """
    data = data.sort_values(by=["content", "skill"])
    data = data.set_index(pd.Series([i for i in range(data.shape[0])]))
    data["true"] = [""] * data.shape[0]

    for i in range(0, data.shape[0], 3):
        sentiment_1 = data.iloc[i].sentiment
        sentiment_2 = data.iloc[i + 1].sentiment
        sentiment_3 = data.iloc[i + 2].sentiment

        weight_1 = data.iloc[i].weight
        weight_2 = data.iloc[i + 1].weight
        weight_3 = data.iloc[i + 2].weight

        sentiments = [sentiment_1, sentiment_2, sentiment_3]
        weights = [weight_1, weight_2, weight_3]
        # positive, negative, neutral
        probs = [0, 0, 0]

        for j in range(3):
            if sentiments[j] == "positive":
                probs[0] += weights[j] / 3
            if sentiments[j] == "negative":
                probs[1] += weights[j] / 3
            if sentiments[j] == "neutral":
                probs[2] += weights[j] / 3

        true_idx = np.argmax(probs)
        if true_idx == 0:
            data.loc[i, "true"] = "positive"
            data.loc[i + 1, "true"] = "positive"
            data.loc[i + 2, "true"] = "positive"
        elif true_idx == 1:
            data.loc[i, "true"] = "negative"
            data.loc[i + 1, "true"] = "negative"
            data.loc[i + 2, "true"] = "negative"
        elif true_idx == 2:
            data.loc[i, "true"] = "neutral"
            data.loc[i + 1, "true"] = "neutral"
            data.loc[i + 2, "true"] = "neutral"

    data = data[data.sentiment == data.true]
    data = data.drop_duplicates(subset=["content"], keep="first")

    return data[["topic", "content", "sentiment"]]
