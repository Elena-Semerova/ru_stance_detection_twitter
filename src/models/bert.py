import random
import re
import string
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(value: int) -> None:
    """
    Fixing seed

    Params:
    -------
        seed (int): seed value
    """
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)


class RussianStanceTwitterDataset(Dataset):
    def __init__(
        self, tweets: np.ndarray, stances: np.ndarray, tokenizer: Any, max_len: int
    ):
        self.tweets = tweets
        self.stances = stances
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        tweet = str(self.tweets[item])
        stance = self.stances[item]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return {
            "tweet_text": tweet,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "stances": torch.tensor(stance, dtype=torch.long),
        }


class RussianStanceTwitterClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super(RussianStanceTwitterClassifier, self).__init__()
        self.bert = model
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(
        self, input_ids: List[int], attention_mask: Optional[torch.FloatTensor]
    ) -> Any:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]

        return self.out(self.drop(last_hidden_state_cls))


def create_train_dataloader(
    X_data: np.ndarray,
    y_data: np.ndarray,
    tokenizer: Any,
    batch_size: int,
    max_len: int,
) -> DataLoader:
    """
    Creating train dataloader

    Params:
    -------
        X_data (np.ndarray): input train data
        y_data (np.ndarray): input train labels
        tokenizer (Any): tokenizer for data
        batch_size (int): size for batches
        max_len (int): max length for vectors

    Returns:
    --------
        (DataLoader): train dataloader
    """
    dataset = RussianStanceTwitterDataset(
        tweets=X_data, stances=y_data, tokenizer=tokenizer, max_len=max_len
    )

    return DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=batch_size,
    )


def create_test_dataloader(
    X_data: np.ndarray, tokenizer: Any, batch_size: int, max_len: int
) -> DataLoader:
    """
    Creating test dataloader

    Params:
    -------
        X_data (np.ndarray): input test data
        tokenizer (Any): tokenizer for data
        batch_size (int): size for batches
        max_len (int): max length for vectors

    Returns:
    --------
        (DataLoader): test dataloader
    """
    dataset = RussianStanceTwitterDataset(
        tweets=X_data, stances=[0] * len(X_data), tokenizer=tokenizer, max_len=max_len
    )

    return DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
    )


def train_epoch(
    model: Any,
    data_loader: DataLoader,
    loss_fn: Any,
    optimizer: Any,
    device: torch.device,
    scheduler: Any,
    n_examples: int,
) -> Tuple[float, float]:
    """
    Training for epoch

    Params:
    -------
        model (Any): model that trained
        data_loader (DataLoader): train dataloader
        loss_fn (Any): loss function
        optimizer (Any): optimizer for training
        device (torch.device): device for trainig
        scheduler (Any): scheduler for trainig
        n_examples (int): count of examples at data

    Returns:
    --------
        (Tuple[float, float]): accuracy and loss
    """
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="TRAIN"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        stances = d["stances"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, stances)
        correct_predictions += torch.sum(preds == stances)

        losses.append(loss.item())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


@torch.no_grad()
def eval_model(
    model: Any,
    data_loader: DataLoader,
    loss_fn: Any,
    device: torch.device,
    n_examples: int,
):
    """
    Validating for epoch

    Params:
    -------
        model (Any): model that trained
        data_loader (DataLoader): validation dataloader
        loss_fn (Any): loss function
        device (torch.device): device for validation
        n_examples (int): count of examples at data

    Returns:
    --------
        (Tuple[float, float]): accuracy and loss
    """
    model = model.eval()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="EVALUATION"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        stances = d["stances"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, stances)
        correct_predictions += torch.sum(preds == stances)

        losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


@torch.no_grad()
def get_predictions(model: Any, data_loader: DataLoader) -> Tuple[Tensor, Tensor]:
    """
    Predicting labels for valid data

    Params:
    -------
        model (Any): trained model
        data_loader (DataLoader): validation dataloader

    Returns:
    --------
        (Tuple[Tensor, Tensor]): predictions and probabilities
    """
    model.eval()

    predictions = []
    prediction_probs = []
    real_values = []

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)

        outputs = F.softmax(model(input_ids=input_ids, attention_mask=attention_mask))
        _, preds = torch.max(outputs, dim=1)

        predictions.extend(preds)
        prediction_probs.extend(outputs)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()

    return predictions, prediction_probs


def include_info_about_topics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Including information about topic like a feature

    Params:
    -------
        data (pd.DataFrame): input data

    Returns:
    --------
        data (pd.DataFrame): data with topic as a feature
    """
    for i in range(data.shape[0]):
        tweet = data.loc[i].content
        topic = data.loc[i].topic

        if topic == "культура отмены":
            tweet_with_topic = "ТЕМА_КУЛЬТУРА_ОТМЕНЫ " + tweet
        elif topic == "феминизм":
            tweet_with_topic = "ТЕМА_ФЕМИНИЗМ " + tweet
        elif topic == "ЛГБТК+":
            tweet_with_topic = "ТЕМА_ЛГБТК+ " + tweet
        elif topic == "эйджизм":
            tweet_with_topic = "ТЕМА_ЭЙДЖИЗМ " + tweet
        elif topic == "лукизм":
            tweet_with_topic = "ТЕМА_ЛУКИЗМ " + tweet

        data.loc[i, "content"] = tweet_with_topic
    return data


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
    train_0 = train_data[train_data.stance == 0]
    train_1 = train_data[train_data.stance == 1]
    train_2 = train_data[train_data.stance == 2]

    minority = min([train_0.shape[0], train_1.shape[0], train_2.shape[0]])

    train_0 = train_0.sample(minority)
    train_1 = train_1.sample(minority)
    train_2 = train_2.sample(minority)

    train_data = pd.concat([train_0, train_1, train_2], ignore_index=True)

    return train_data.sample(frac=1).reset_index(drop=True)


def training(
    data: pd.DataFrame,
    batch_size: int,
    epochs: int,
    learning_rate_optimizer: float,
    n_classes: int,
    include_topics: bool = True,
    balancing: str = None,
) -> None:
    """
    Training model using BERT

    Params:
    -------
        data (pd.DataFrame): input data
        batch_size (int): size for batches
        epochs (int): number of epochs
        learning_rate_optimizer (float): learning rate for optimizer
        n_classes (int): count of classes
        include_topics (bool): flag for including topic as a feature
        balancing (str): flag for balancing data
    """
    seed(42)

    if include_topics:
        data = include_info_about_topics(data)

    sample = data.sample(frac=1)
    train_data = sample.iloc[data.shape[0] // 100 * 8 :]
    valid_data = sample.iloc[: data.shape[0] // 100 * 8]

    if balancing == "all":
        train_data = balancing_data(train_data)
    elif balancing == "each":
        train_data_cc = train_data[train_data.topic == "культура отмены"]
        train_data_fem = train_data[train_data.topic == "феминизм"]
        train_data_lgbt = train_data[train_data.topic == "ЛГБТК+"]
        train_data_age = train_data[train_data.topic == "эйджизм"]
        train_data_look = train_data[train_data.topic == "лукизм"]

        train_data_cc = balancing_data(train_data_cc)
        train_data_fem = balancing_data(train_data_fem)
        train_data_lgbt = balancing_data(train_data_lgbt)
        train_data_age = balancing_data(train_data_age)
        train_data_look = balancing_data(train_data_look)

        train_data = pd.concat(
            [
                train_data_cc,
                train_data_fem,
                train_data_lgbt,
                train_data_age,
                train_data_look,
            ],
            ignore_index=True,
        )

        train_data = train_data.sample(frac=1).reset_index(drop=True)

    X_train = train_data.content.values
    y_train = train_data.stance.values

    X_valid = valid_data.content.values
    y_valid = valid_data.stance.values

    train_tokenized = [tokenizer.encode(x, add_special_tokens=True) for x in X_train]
    valid_tokenized = [tokenizer.encode(x, add_special_tokens=True) for x in X_valid]

    train_max_len = max(map(len, train_tokenized))
    valid_max_len = max(map(len, valid_tokenized))

    print("MAX_LEN_train:\t", train_max_len)
    print("MAX_LEN_valid:\t", valid_max_len)
    print()

    train_data_loader = create_train_dataloader(
        X_train, y_train, tokenizer, batch_size, train_max_len
    )
    valid_data_loader = create_test_dataloader(
        X_valid, tokenizer, batch_size, valid_max_len
    )

    model = RussianStanceTwitterClassifier(n_classes)
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=learning_rate_optimizer, correct_bias=False
    )

    total_steps = len(train_data_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1:2d}/{epochs:2d}")
        print("-" * 25)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(X_train),
        )
        valid_acc, valid_loss = eval_model(
            model, valid_data_loader, loss_fn, device, len(X_valid)
        )

        print(f"Train loss {train_loss:.4f} accuracy {train_acc:.4f}")
        print(f"Valid loss {valid_loss:.4f} accuracy {valid_acc:.4f}")

    predicted_valid_labels, prediction_probs_valid = get_predictions(
        model, valid_data_loader
    )
    print(
        classification_report(
            y_valid,
            predicted_valid_labels,
            target_names=["against", "favor", "neutral"],
        )
    )
