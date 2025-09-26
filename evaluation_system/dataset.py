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
from typing import List, Dict


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
        context = row['context']
        context_speakers = row['context_speakers']
        context_formatted = ''

        for text, speaker in zip(context, context_speakers):
            context_formatted = f'{speaker}: {text}\n'

        data.append({
            'id': row['id'],
            'text': utterance,
            'label': 1 if sarcasm else 0,
            'context': context_formatted
        })

    formatted_df = pd.DataFrame(data)
    return formatted_df
