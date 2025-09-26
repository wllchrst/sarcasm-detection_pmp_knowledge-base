from abc import ABC, abstractmethod
from typing import Dict
class BasePrompt(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_prompt(self) -> Dict[str, str]:
        pass
    