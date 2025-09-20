from interfaces import LLMType
from llm import GeminiLLM, OllamaLLM, BaseLLM
from typing import Optional, List, TypedDict
from transformers import pipeline


class SentimentAnalysis(TypedDict):
    label: str
    score: float


class NERProcessor:
    def __init__(self, llm_type: LLMType, model_name: Optional[str] = None):
        self.llm = self.initialize_llm(llm_type, model_name)
        self.pipe = pipeline("sentiment-analysis",
                             model="finiteautomata/bertweet-base-sentiment-analysis")

        # finiteautomata/bertweet-base-sentiment-analysis

    def initialize_llm(self, llm_type: LLMType, model_name: Optional[str]):
        if llm_type == LLMType.GEMINI:
            return GeminiLLM()

        elif llm_type == LLMType.OLLAMA:
            if model_name is None:
                raise ValueError("Model name cannot be None if llm type is OLLAMA")
            return OllamaLLM(model_name)

    def process_verbs(self, verbs: List[str], original_sentence: str) -> List[SentimentAnalysis]:
        sentiments: List[SentimentAnalysis] = []
        for verb in verbs:
            if verb not in original_sentence:
                continue

            result = self.pipe(verb)
            analysis = SentimentAnalysis(**result[0])
            sentiments.append(analysis)

        return sentiments

    def get_word_information(self, words: List[str]) -> List[str]:
        informations = []
        for word in words:
            system_prompt = f'Really Short and Compact information of the word I\'m going to give in one paragraph'
            result = self.llm.answer(system_prompt=system_prompt, prompt=word)
            informations.append(result)
        
        return informations