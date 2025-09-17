from abc import ABC, abstractmethod
class BasePromptHandler(ABC):
    @abstractmethod
    def get_response(self, text: str) -> int:
        pass
    
    @abstractmethod
    def process_response(self, response: str) -> int:
        pass
    
