import pandas as pd
import os
from ner.ner_processor import NERProcessor
from evaluation_system.dataset import load_twitter_indonesian_dataset
from interfaces import LLMType

OUTPUT_FOLDER = 'twitter_with_context'

def load_objects():
    processor = NERProcessor(
        llm_type=LLMType.OLLAMA,
        use_wiki=False,
        sentiment_model='bert_tweet',
        with_logging=False,
        model_name='qwen3:8b'
    )
    train_df = load_twitter_indonesian_dataset('train')
    validation_df = load_twitter_indonesian_dataset('validation')
    test_df = load_twitter_indonesian_dataset('test')

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
    saved_dataframe = None
    if os.path.exists(file_path):
        saved_dataframe = pd.read_csv(file_path)

    for _, row in dataframe.iterrows():
        text = row['text']
        words = processor.get_unknown_words(text, unknown_words)
        unknown_words.append(words)
    
    dataframe = dataframe.iloc[:len(unknown_words)].copy()
    dataframe['unknown_words'] = unknown_words

    return dataframe

def generate_output_path(partition: str):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


    return f'{OUTPUT_FOLDER}/{partition}.csv'
    

def start_generate_context():
    processor, dataframe_dictionary = load_objects()

    for partition, df in dataframe_dictionary.items():
        print(f"Processing partition: {partition}")

        df_with_unknowns = detect_unknown_words(processor, df)
        output_path = generate_output_path(partition)
        df_with_unknowns.to_csv(output_path, index=False)

        print(f"Saved processed data with unknown words to {output_path}")


if __name__ == "__main__":
    start_generate_context()