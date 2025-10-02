import pandas as pd
import os
from ner.ner_processor import NERProcessor
from evaluation_system.dataset import load_twitter_indonesian_dataset
from interfaces import LLMType
from typing import Optional
from helpers.argument_helper import ArgumentHelper

OUTPUT_FOLDER = 'twitter_with_context'


def load_objects():
    processor = NERProcessor(
        llm_type=LLMType.OLLAMA,
        use_wiki=False,
        sentiment_model='bert_tweet',
        with_logging=False,
        model_name='qwen3:8b'
    )
    train_df = load_twitter_indonesian_dataset(partition='train')
    validation_df = load_twitter_indonesian_dataset(partition='validation')
    test_df = load_twitter_indonesian_dataset(partition='test')

    dataframe_dict = {
        'train': train_df,
        'validation': validation_df,
        'test': test_df
    }

    return processor, dataframe_dict


def detect_unknown_words(processor: NERProcessor,
                         dataframe: pd.DataFrame,
                         file_path: str = None) -> pd.DataFrame:
    unknown_words = []
    texts = []
    labels = []
    ids = []

    saved_dataframe = None
    if os.path.exists(file_path):
        saved_dataframe = pd.read_csv(file_path)

    predicted_ids = [] if saved_dataframe is None else saved_dataframe['id'].values

    print(f'predicted_ids {predicted_ids}')

    try:
        for index, row in dataframe.iterrows():
            print(index)
            current_id = row['id']
            text = row['text']
            label = row['label']

            ids.append(current_id)
            texts.append(text)
            labels.append(label)

            if current_id in predicted_ids:
                print(f'Skipping id {current_id}')
                words = saved_dataframe.loc[saved_dataframe['id'] == current_id, 'unknown_words'].iloc[0]
                unknown_words.append(words)
            else:
                words = processor.get_unknown_words(text=text)
                unknown_words.append(words)

    except Exception as e:
        checkpoint_df = pd.DataFrame({
            'id': ids,
            'texts': texts,
            'label': labels,
            'unknown_words': unknown_words
        })

        checkpoint_df.to_csv(file_path, index=False)
        raise e

    return pd.DataFrame({
        'id': ids,
        'texts': texts,
        'label': labels,
        'unknown_words': unknown_words
    })


def generate_output_path(partition: str):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    return f'{OUTPUT_FOLDER}/{partition}.csv'


def start_generate_context(partition: Optional[str] = None):
    processor, dataframe_dictionary = load_objects()

    if partition is not None:
        if partition in dataframe_dictionary:
            dataframe_dictionary = {
                partition: dataframe_dictionary[partition]
            }
        else:
            raise ValueError(f"Partition '{partition}' not found in dataframe_dictionary")

    for partition, df in dataframe_dictionary.items():
        print(f"Processing partition: {partition}")

        output_path = generate_output_path(partition)
        df_with_unknowns = detect_unknown_words(
            dataframe=df,
            processor=processor,
            file_path=output_path
        )

        df_with_unknowns.to_csv(output_path, index=False)

        print(f"Saved processed data with unknown words to {output_path}")


if __name__ == "__main__":
    arguments = ArgumentHelper.parse_context_generation()

    if arguments.partition is None:
        raise ValueError("arguments.partition cannot be none")
    start_generate_context(partition=arguments.partition)
