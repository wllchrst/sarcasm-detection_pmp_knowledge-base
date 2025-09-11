from interfaces import LLMType
from llm import GeminiLLM, OllamaLLM, BaseLLM
from typing import Optional, List
from transformers import pipeline


class NERProcessor:
    def __init__(self, llm_type: LLMType, model_name: Optional[str]):
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

    def process_verbs(self, verbs: List[str], original_sentence: str):
        for verb in verbs:
            result = self.pipe(verb)
            print(f'{verb}: {result}')
