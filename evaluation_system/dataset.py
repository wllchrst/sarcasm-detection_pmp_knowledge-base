"""
Make sure all dataset that is loaded from this file is already processed by labeling the column name

Also please make sure dataset that is loaded has id column to for saving checkpoint the result if there is an error.

Label Column Name: 'label'
Text Column Name: 'text'
Context Column Name: 'context'
ID Column Name: 'id'
"""
import pandas as pd
import json
import re
from typing import List, Dict
from datasets import load_dataset


def load_semeval_dataset(file_path: str = 'SemEval2018-T3_gold_test_taskA_emoji.txt') -> pd.DataFrame:
    semeval_df = pd.read_csv(file_path, sep='\t')

    return semeval_df.rename(columns={'Tweet text': 'text', 'Label': 'label', 'Tweet index': 'id'})


def load_mustard_dataset(file_path: str = 'mustard_sarcasm_data.json'):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data is None:
        raise Exception("Error getting data")

    df = pd.DataFrame.from_dict(data, orient="index").reset_index()

    df = df.rename(columns={"index": "id"})
    print(df.head())
    print(df.columns)

    data: List[Dict] = []
    for _, row in df.iterrows():
        sarcasm = row['sarcasm']
        utterance = row['utterance']
        speakers = row['speaker']
        context = row['context']
        context_speakers = row['context_speakers']

        context_formatted = ''
        for text, context_speaker in zip(context, context_speakers):
            context_formatted += f'{context_speaker}: {text} '

        speakers_formatted = f'{speakers}: {utterance}'

        text_formatted = f"{context_formatted}{{{speakers_formatted}}}"

        data.append({
            'id': row['id'],
            'text': text_formatted,
            'label': 1 if sarcasm else 0
        })

    formatted_df = pd.DataFrame(data)
    return formatted_df


def remove_angle_brackets(text: str) -> str:
    """
    Remove all substrings enclosed in < >, e.g., <username>, <link>.
    Also collapses multiple spaces into one.
    """
    cleaned = re.sub(r"<.*?>", "", text)  # remove <...>
    cleaned = re.sub(r"\s+", " ", cleaned)  # normalize spaces

    return cleaned.strip()


def add_partition_id_column(df: pd.DataFrame, partition: str) -> pd.DataFrame:
    """
    Add an 'id' column to the dataframe where each row is assigned
    a unique identifier with the partition name as prefix.

    Example:
        partition="test" → test_0, test_1, ...
    """
    df = df.copy()
    df["id"] = [f"{partition}_{i}" for i in range(len(df))]
    return df


def load_twitter_indonesian_dataset(
        hugging_face_link: str = 'w11wo/twitter_indonesia_sarcastic',
        partition: str = 'test'
) -> pd.DataFrame:
    dataset = load_dataset(hugging_face_link)

    if partition != 'all':
        dataframe = dataset[partition].to_pandas()
    else:
        dataframes = [ds.to_pandas() for ds in dataset.values()]
        dataframe = pd.concat(dataframes, ignore_index=True)

    if "tweet" in dataframe.columns:
        dataframe["tweet"] = dataframe["tweet"].apply(remove_angle_brackets)

    dataframe = add_partition_id_column(dataframe, partition)

    return dataframe.rename(columns={
        'tweet': 'text',
        'label': 'label'
    })
