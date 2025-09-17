from abc import ABC, abstractmethod
class BasePrompt(ABC):
    @abstractmethod
    def generate_prompt(self, text: str) -> str:
        pass
