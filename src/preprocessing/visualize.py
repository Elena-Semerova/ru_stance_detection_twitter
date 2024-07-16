import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")


def make_examples(data: pd.DataFrame, size: int = 5) -> None:
    """
    Printing some examples of tweets by input dataframe

    Params:
    -------
        data (pd.DataFrame): input dataframe
        size (int): count of examples
    """
    for tweet in data.sample(size).content.values:
        print(tweet)
        print()


def make_date_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Making date features at dataframe

    Params:
    -------
        data (pd.DataFrame): input dataframe

    Returns:
    --------
        data (pd.DataFrame): datsframe with date features
    """
    data["date"] = pd.to_datetime(data["date"])
    data["day"] = data["date"].dt.strftime("%d")
    data["month"] = data["date"].dt.strftime("%m")
    data["year"] = data["date"].dt.strftime("%y")
    data["year_month"] = data["date"].dt.strftime("%y-%m")

    return data


def visualize_year_month(data: pd.DataFrame, topic_name: str = "") -> None:
    """
    Visualizing dependence of number of tweets by month and year for topic

    Params:
    -------
        data (pd.DataFrame): input dataframe
        topic_name (str): dataframe's topic
    """
    data_date = make_date_features(data)
    df_groupby = data_date.groupby(["year_month"]).count()["content"]
    df_count = pd.DataFrame(
        {"year_month": data_date.year_month.unique(), "count": df_groupby.values}
    )

    sns.relplot(
        data=df_count, x="year_month", y="count", kind="line", height=5, aspect=3
    )

    plt.title(
        "Dependence of number of tweets on the year_month for topic: " + topic_name,
        size=15,
    )
    plt.xlabel("year_month", fontsize=12)
    plt.ylabel("number of tweets", fontsize=12)

    plt.show()
