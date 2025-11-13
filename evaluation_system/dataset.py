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
import ast
import os

from typing import List, Dict
from datasets import load_dataset
from matplotlib import pyplot as plt

PARTITION_LIST = ['train', 'test', 'validation']
SAVE_PLOT_FOLDER = 'dataset_information'


def load_semeval_dataset(file_path: str = 'SemEval2018-T3_gold_test_taskA_emoji.txt',
                         information_file_path: str = 'twitter_with_context/information_semeval.csv',
                         detected_file_path: str = 'twitter_with_context/semeval/test.csv') -> pd.DataFrame:
    if not os.path.exists(information_file_path):
        semeval_df = pd.read_csv(file_path, sep='\t')
        return semeval_df.rename(columns={'Tweet text': 'text', 'Label': 'label', 'Tweet index': 'id'})

    information_df = pd.read_csv(information_file_path)
    detected_df = pd.read_csv(detected_file_path)

    definitions = []

    for index, row in detected_df.iterrows():
        unknowns = ast.literal_eval(row['unknown_words'])
        context_formatted = 'Definition for keywords:\n\n'

        for word in unknowns:
            word = word.lower()
            if word == '':
                continue

            find_definitions = information_df.loc[information_df['word'] == word, 'definition']

            if len(find_definitions) == 0:
                print(f'Unknown definition for: {word}')
                definition = ''
            else:
                definition = find_definitions.iloc[0]

            context_formatted += f'{definition}\n'

        definitions.append(context_formatted if len(unknowns) > 0 else '')

    detected_df['context'] = definitions
    return detected_df.rename(columns={
        'id': 'id',
        'texts': 'text',
        'label': 'label',
        'context': 'context'
    })


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
            'label': 1 if sarcasm else 0,
            'speaker': speakers,
            'context_speakers': context_speakers,
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


def load_twitter_indonesian_dataset_for_evaluation(folder_path: str = 'twitter_with_context',
                                                   partition: str = 'test'):
    information_df = pd.read_csv(f'{folder_path}/information.csv')
    print(information_df.head())

    def construct_dataset(dataframe: pd.DataFrame, information_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct dataset by joining the dataframe without list of information with the information df.

        :param dataframe: main dataset source
        :param information_df: dataset pure for the word definition
        :return: Constructed Dataset
        """
        definitions: List[str] = []

        for index, row in dataframe.iterrows():
            unknown_words = ast.literal_eval(row['unknown_words'])
            if len(unknown_words) <= 0:
                continue
            context_formatted = 'Definisi kata-kata penting:\n\n'

            for word in unknown_words:
                word = word.lower()
                if word == '':
                    continue
                find_definitions = information_df.loc[information_df['word'] == word, 'definition']

                if len(find_definitions) == 0:
                    print(f'Uknown definition for: {word}')
                    definition = ''
                else:
                    definition = find_definitions.iloc[0]

                context_formatted += f'{definition}\n'

            definitions.append(context_formatted)

        dataframe['context'] = definitions
        return dataframe.rename(columns={
            'id': 'id',
            'texts': 'text',
            'label': 'label',
            'context': 'context'
        })

    joined_df = None

    for p in PARTITION_LIST:
        df = pd.read_csv(f'{folder_path}/{partition}.csv')
        if p == partition:
            return construct_dataset(dataframe=df, information_df=information_df)
        elif partition == 'all':
            joined_df = df if joined_df is None else pd.concat([joined_df, df])

    if partition == 'all':
        return construct_dataset(dataframe=joined_df,
                                 information_df=information_df)

    raise ValueError('Partition value is wrong')


def generate_plot():
    os.makedirs(SAVE_PLOT_FOLDER, exist_ok=True)

    # 🔹 Set all font sizes globally
    plt.rcParams.update({
        'font.size': 22,  # default font size for all text
        'axes.titlesize': 22,  # title size
        'axes.labelsize': 22,  # x/y label size
        'xtick.labelsize': 22,  # x-axis tick labels
        'ytick.labelsize': 22,  # y-axis tick labels
        'legend.fontsize': 22,  # legend text
    })

    datasets = {
        'semeval': load_semeval_dataset(),
        'mustard': load_mustard_dataset(),
        'twitter_indo': load_twitter_indonesian_dataset()
    }

    label_mapping = {1: "sarcasm", 0: "not sarcasm"}
    label_order = ["not sarcasm", "sarcasm"]
    colors = ["#3498db", "#e74c3c"]  # vibrant blue & red

    for key, dataset in datasets.items():
        dataset['label'] = dataset['label'].map(label_mapping)

        label_counts = dataset['label'].value_counts()
        label_counts = label_counts.reindex(label_order)

        plt.figure(figsize=(8, 7))
        ax = label_counts.plot(kind="bar", color=colors)

        # Title & axes labels
        plt.title(f"Label Distribution for {key.upper()} Dataset")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.ylim(0, max(label_counts) * 1.15)

        # Add count labels on top of each bar
        for i, count in enumerate(label_counts):
            ax.text(i, count + max(label_counts) * 0.02, str(count),
                    ha='center', fontsize=22, fontweight='bold')

        plt.tight_layout()

        save_path = os.path.join(SAVE_PLOT_FOLDER, f"{key}_label_distribution.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved plot for {key} dataset at {save_path}")
