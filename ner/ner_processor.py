from helpers import WordHelper
from interfaces import LLMType
from llm import GeminiLLM, OllamaLLM, BaseLLM
from typing import Optional, List, TypedDict
from transformers import pipeline
from es import EsRetriever


class SentimentAnalysis(TypedDict):
    label: str
    score: float


class NERProcessor:
    def __init__(self, llm_type: LLMType, use_wiki: bool, sentiment_model: str, with_logging: bool,
                 model_name: Optional[str] = None):
        self.with_logging = with_logging
        self.llm = self.initialize_llm(llm_type, model_name)
        if sentiment_model:
            self.pipe = self.load_sentiment(sentiment_model)
        if use_wiki == True:
            self.es_retriever = EsRetriever()

    def load_sentiment(self, sentiment_model: str):
        if sentiment_model == 'bert_tweet':
            return pipeline("sentiment-analysis",
                            model="finiteautomata/bertweet-base-sentiment-analysis")

        raise ValueError(f"Sentiment model {sentiment_model} is not in the list of sentiment model that can be use")

    def initialize_llm(self, llm_type: LLMType, model_name: Optional[str]):
        if llm_type == LLMType.GEMINI:
            return GeminiLLM()

        elif llm_type == LLMType.OLLAMA:
            if model_name is None:
                raise ValueError("Model name cannot be None if llm type is OLLAMA")
            return OllamaLLM(model_name)

    def process_verbs(self, verbs: List[str], original_sentence: str) -> List[SentimentAnalysis]:
        sentiments: List[SentimentAnalysis] = []
        unique_verbs = set(verbs)
        for verb in unique_verbs:
            if verb not in original_sentence:
                continue

            result = self.pipe(verb)
            analysis = SentimentAnalysis(**result[0])
            sentiments.append(analysis)

        return sentiments

    def get_word_information(self, words: List[str], use_wiki: bool) -> List[str]:
        information = []
        unique_words = set(word.strip() for word in words)
        for word in unique_words:
            word = word.strip()
            info = self.wiki_info(word) if use_wiki else self.llm_info(word)
            information.append(info)

        return information

    def llm_info(self, word: str) -> str:
        system_prompt = f'Provide very short and compact 1 paragraph information about the word: {word}'
        result = self.llm.answer(system_prompt=system_prompt, prompt=word, with_logging=self.with_logging)
        return result

    def wiki_info(self, word: str) -> str:
        documents = self.es_retriever.search_wiki_data(
            index='wikipedia',
            total_result=1,
            query=word
        )

        if len(documents) == 0:
            return '-'

        document = documents[0]
        return document['text']

    def get_unknown_words(self, text: str):
        system_prompt = 'Anda akan diberikan sebuah teks dari twitter, tugas anda adalah untuk memberikan kata yang anda tidak mengerti dari teks yang diberikan'
        response = self.llm.answer(
            system_prompt=system_prompt,
            prompt=text,
            with_logging=False)

        system_prompt = (
                "Anda akan diberikan teks berisi penjelasan kata-kata yang tidak dimengerti\n"
                + "Tugas anda adalah untuk memisahkan kata-kata tersebut menjadi commma separated values (csv)\n"
                + "contoh:\n"
                + "pertama,kedua,ketiga"
        )

        formatted_response = self.llm.answer(
            system_prompt=system_prompt,
            prompt=response,
            with_logging=False
        )

        words = formatted_response.split(',')
        final_words = []

        for word in words:
            word = WordHelper.remove_non_alphabetic(word)
            word = WordHelper.normalize_repeated_chars(word)
            final_words.append(word)

        return final_words

    def get_sentence_context_full_llm(self, sentence: str):
        words = self.get_unknown_words(sentence)
        contexts = []

        for word in words:
            break

        raise Exception("Testing")
