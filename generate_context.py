import pandas as pd
import os
import requests
import re
import traceback
import ast

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from ner.ner_processor import NERProcessor
from evaluation_system.dataset import load_twitter_indonesian_dataset
from interfaces import LLMType
from typing import Optional
from helpers.argument_helper import ArgumentHelper
from helpers import env_helper, WordHelper
from joblib import Memory
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm import OllamaLLM

memory = Memory("cache_dir", verbose=0)
OUTPUT_FOLDER = 'twitter_with_context'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/129.0.0.0 Safari/537.36"
}
INFORMATION_DATASET_FILENAME = 'information.csv'


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

            if current_id in predicted_ids:
                print(f'Skipping id {current_id}')
                words = saved_dataframe.loc[saved_dataframe['id'] == current_id, 'unknown_words'].iloc[0]
                unknown_words.append(words)
            else:
                words = processor.get_unknown_words(text=text)
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


# @memory.cache
def retrieve_relevant_website(word: str) -> List[str]:
    try:
        links = []
        url = 'https://www.googleapis.com/customsearch/v1'
        params = {
            'q': f'Apa itu {word}',
            'key': env_helper.GOOGLE_SEARCH_API_KEY,
            'cx': env_helper.SEARCH_ENGINE_ID
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        response = response.json()

        if 'items' not in response:
            return []

        for item in response['items']:
            link = item['link']
            links.append(link)

            if len(links) > 3:
                return links

        return links
    except Exception as e:
        print(f'Error retrieving relevant website for {word}: {e}')

        if 'Too Many Requests' in str(e):
            raise e

        return []


def load_text_from_web(web_content: str, key_word: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Splits the web content into chunks, calculates BM25 scores based on the query "arti {key_word}",
    and returns the top-k most relevant chunks with their BM25 scores.

    Args:
        web_content (str): Raw text extracted from a website.
        key_word (str): The keyword you want to find meaning for.
        top_k (int): Number of top relevant chunks to return.

    Returns:
        list[tuple[str, float]]: List of (chunk, bm25_score) sorted by relevance.
    """
    if web_content.strip() == '':
        return []

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=10,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_text(web_content)
        tokenized_corpus = [
            re.findall(r"\w+", chunk.lower()) for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_corpus)
        query = re.findall(r"\w+", f"arti {key_word.lower()}")
        scores = bm25.get_scores(query)

        ranked_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked_chunks[:top_k]

    except Exception as e:
        print(f"Error during BM25 ranking: {e}")
        raise


def retrieve_important_chunks(links: List[str], key_word: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Retrieves and ranks important text chunks from multiple web pages using BM25.

    Args:
        links (List[str]): List of URLs to fetch content from.
        key_word (str): Keyword to focus on (e.g., "tolol").
        top_k (int): Number of top chunks to return overall.

    Returns:
        List[Tuple[str, float]]: List of (chunk, bm25_score) sorted by global relevance.
    """
    retrieved_chunks: List[Tuple[str, float]] = []

    for link in links:
        try:
            response = None
            try:
                response = requests.get(link, headers=headers, timeout=10)
                response.raise_for_status()
            except Exception as e:
                print(f'Error fetching {link}: {e}')

            if response is None:
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            web_content = soup.get_text()

            chunks = load_text_from_web(
                web_content=web_content,
                key_word=key_word,
                top_k=top_k
            )

            retrieved_chunks.extend(chunks)

        except Exception as e:
            raise e

    sorted_chunks = sorted(retrieved_chunks, key=lambda x: x[1], reverse=True)
    return sorted_chunks[:top_k]


def conclude_retrieved_information(llm: OllamaLLM, word: str, informations: Tuple[str, float]) -> str:
    formatted_information = 'Informasi:\n\n'
    for info in informations:
        sentence, _ = info
        formatted_information += f'-{WordHelper.replace_enters_with_space(sentence)}\n'

    system_prompt = (
            'Anda akan diberikan beberapa kalimat yang merupakan informasi, kalimat ini bertujuan untuk memberikan definisi terhadap suatu kata.\n'
            + 'Tugas anda adalah untuk memberikan definisi terhadap kata tersebut berdasarkan informasi yang ada\n\n'
            + 'Pastikan jawaban anda sederhana dan tidak terlalu panjang'
    )

    prompt = (
            formatted_information +
            "\n\n" +
            f'Kata yang perlu di definisi: {word}'
    )

    return llm.answer(system_prompt=system_prompt,
                      prompt=prompt, with_logging=False)


def get_word_definition(llm: OllamaLLM, word: str):
    links = retrieve_relevant_website(word)
    if len(links) == 0:
        return ''

    chunks = retrieve_important_chunks(links, key_word=word)

    if len(chunks) == 0:
        return ''

    return conclude_retrieved_information(
        llm=llm,
        word=word,
        informations=chunks
    )


def generate_output_path(partition: str):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    return f'{OUTPUT_FOLDER}/{partition}.csv'


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


def start_generate_context(partition: Optional[str] = None):
    processor, dataframe_dictionary = load_objects()

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

        output_path = generate_output_path(partition)
        df_with_unknowns = detect_unknown_words(
            dataframe=df,
            processor=processor,
            file_path=output_path
        )

        df_with_unknowns.to_csv(output_path, index=False)

        print(f"Saved processed data with unknown words to {output_path}")

    # ====================== Generate information for every word ======================
    for partition, df in dataframe_dictionary.items():
        print(f'Generating information for {partition}')
        dataframe_path = generate_output_path(partition)
        dataframe = pd.read_csv(dataframe_path)

        generate_information(processor, dataframe)

        print(f'Finished Generating information for {partition}')


if __name__ == "__main__":
    arguments = ArgumentHelper.parse_context_generation()

    if arguments.partition is None:
        raise ValueError("arguments.partition cannot be none")

    start_generate_context(partition=arguments.partition)
