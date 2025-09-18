from abc import ABC, abstractmethod

class BaseLLM(ABC):
    def __init__(self, llm_model: str = None):
        self.llm_model = llm_model
        super().__init__()
    
    @abstractmethod
    def answer(self, query: str):
        pass