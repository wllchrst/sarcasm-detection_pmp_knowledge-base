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


def retrieve_relevant_website(word: str, is_indonesian: bool) -> List[str]:
    global current_api_index
    query = f'Apa itu {word}' if is_indonesian else word
    try:
        links = []
        url = 'https://www.googleapis.com/customsearch/v1'
        params = {
            'q': query,
            'key': api_keys[current_api_index],
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

        if 'Too Many Requests' in str(e) and current_api_index == len(api_keys) - 1:
            raise e
        elif 'too many requests'.lower() in str(e).lower():
            print(f'switch api index {current_api_index}')
            current_api_index += 1

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
        query = re.findall(r"\w+", f"{key_word.lower()}")
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


def conclude_retrieved_information(llm: OllamaLLM,
                                   word: str,
                                   informations: Tuple[str, float],
                                   is_indonesian: bool) -> str:
    formatted_information = "Informasi:\n\n" if is_indonesian else "Information:\n\n"
    for info in informations:
        sentence, _ = info
        formatted_information += f"- {WordHelper.replace_enters_with_space(sentence)}\n"

    if is_indonesian:
        system_prompt = (
            "Anda akan diberikan beberapa kalimat yang merupakan informasi, "
            "kalimat ini bertujuan untuk memberikan definisi terhadap suatu kata.\n"
            "Tugas anda adalah untuk memberikan definisi terhadap kata tersebut "
            "berdasarkan informasi yang ada.\n\n"
            "Pastikan jawaban anda sederhana dan tidak terlalu panjang."
        )
        prompt = (
                formatted_information +
                "\n\n" +
                f"Kata yang perlu didefinisi: {word}"
        )
    else:
        system_prompt = (
            "You will be given several sentences that provide information. "
            "These sentences aim to describe the meaning of a word.\n"
            "Your task is to provide a clear definition of the word based on the given information.\n\n"
            "Make sure your answer is simple and not too long."
        )
        prompt = (
                formatted_information +
                "\n\n" +
                f"Word to define: {word}"
        )

    answer = llm.answer(system_prompt=system_prompt,
                        prompt=prompt, with_logging=False)

    return answer


def get_word_definition(llm: OllamaLLM, word: str, is_indonesian: bool):
    links = retrieve_relevant_website(word, is_indonesian=is_indonesian)
    if len(links) == 0:
        print(f'got 0 links for {word}')
        return ''

    chunks = retrieve_important_chunks(links, key_word=word)

    if len(chunks) == 0:
        return ''

    return conclude_retrieved_information(
        llm=llm,
        word=word,
        informations=chunks,
        is_indonesian=is_indonesian,
    )
