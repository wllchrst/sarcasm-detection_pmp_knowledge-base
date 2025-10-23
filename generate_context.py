import pandas as pd
import os
import traceback
import ast

from ner.ner_processor import NERProcessor
from evaluation_system.dataset import load_twitter_indonesian_dataset, load_semeval_dataset, load_mustard_dataset
from interfaces import LLMType
from typing import Optional
from helpers.argument_helper import ArgumentHelper
from joblib import Memory
from context_retrieval import get_word_definition

memory = Memory("cache_dir", verbose=0)
OUTPUT_FOLDER = 'twitter_with_context'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/129.0.0.0 Safari/537.36"
}
INFORMATION_DATASET_FILENAME = 'information.csv'


def load_objects(dataset: str):
    processor = NERProcessor(
        llm_type=LLMType.OLLAMA,
        use_wiki=False,
        sentiment_model='bert_tweet',
        with_logging=False,
        model_name='qwen3:8b'
    )

    train_df = None
    validation_df = None
    test_df = None
    is_indonesian = True

    if dataset == 'twitter_indo':
        train_df = load_twitter_indonesian_dataset(partition='train')
        validation_df = load_twitter_indonesian_dataset(partition='validation')
        test_df = load_twitter_indonesian_dataset(partition='test')
    elif dataset == 'semeval':
        test_df = load_semeval_dataset()
        is_indonesian = False
    elif dataset == 'mustard':
        test_df = load_mustard_dataset()
        is_indonesian = False

    dataframe_dict = {
        'train': train_df,
        'validation': validation_df,
        'test': test_df
    }

    return processor, dataframe_dict, is_indonesian


def detect_unknown_words(processor: NERProcessor,
                         is_indonesian: bool,
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

            if current_id in predicted_ids:
                print(f'Skipping id {current_id}')
                words = saved_dataframe.loc[saved_dataframe['id'] == current_id, 'unknown_words'].iloc[0]
                unknown_words.append(words)
            else:
                words = processor.get_unknown_words(text=text, is_indonesian=is_indonesian)
                print(f'{text}: {words}')
                unknown_words.append(words)

            ids.append(current_id)
            texts.append(text)
            labels.append(label)

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


def generate_output_path(partition: str, dataset: str):
    folder = f'{OUTPUT_FOLDER}/{dataset}'
    os.makedirs(folder, exist_ok=True)

    return f'{folder}/{partition}.csv'


def generate_information(process: NERProcessor, dataframe: pd.DataFrame):
    information_filepath = f'{OUTPUT_FOLDER}/{INFORMATION_DATASET_FILENAME}'
    information_dataframe = None
    if os.path.exists(information_filepath):
        information_dataframe = pd.read_csv(information_filepath)

    word_list = information_dataframe['word'].values if information_dataframe is not None else []
    words = []
    definitions = []
    try:
        for index, row in dataframe.iterrows():
            unknown_words = ast.literal_eval(row['unknown_words'])
            for word in unknown_words:
                print(word)
                word = word.lower()
                if word.strip() == '':
                    continue

                if word in word_list:
                    definition = information_dataframe.loc[information_dataframe['word'] == word, 'definition'].iloc[0]
                    definitions.append(definition)
                else:
                    definition = get_word_definition(process.llm, word)
                    definitions.append(definition)
                words.append(word)

    except Exception as e:
        traceback.print_exc()
        print(f'Error generating information: saving information dataset')

    df = pd.DataFrame({
        'word': words,
        'definition': definitions
    })

    df.to_csv(information_filepath, index=False)


def start_generate_context(partition: Optional[str] = None, dataset: str = 'twitter_indo'):
    processor, dataframe_dictionary, is_indonesian = load_objects(dataset=dataset)

    if partition is not None:
        if partition in dataframe_dictionary:
            dataframe_dictionary = {
                partition: dataframe_dictionary[partition]
            }
        else:
            raise ValueError(f"Partition '{partition}' not found in dataframe_dictionary")

    # ====================== Get unknown words for the dataset ======================
    for partition, df in dataframe_dictionary.items():
        print(f"Processing partition: {partition}")

        output_path = generate_output_path(partition, dataset=dataset)
        df_with_unknowns = detect_unknown_words(
            dataframe=df,
            is_indonesian=is_indonesian,
            processor=processor,
            file_path=output_path
        )

        df_with_unknowns.to_csv(output_path, index=False)

        print(f"Saved processed data with unknown words to {output_path}")

    # ====================== Generate information for every word ======================
    for partition, df in dataframe_dictionary.items():
        print(f'Generating information for {partition}')
        dataframe_path = generate_output_path(partition, dataset=dataset)
        dataframe = pd.read_csv(dataframe_path)

        generate_information(processor, dataframe)

        print(f'Finished Generating information for {partition}')


if __name__ == "__main__":
    arguments = ArgumentHelper.parse_context_generation()
    print(f'Running generation script using | arguments:\n{arguments}')

    if arguments.partition is None:
        raise ValueError("arguments.partition cannot be none")

    start_generate_context(partition=arguments.partition,
                           dataset=arguments.dataset)
