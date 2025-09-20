from prompt import BasePrompt
from typing import List

class NERPrompt(BasePrompt):
    def __init__(self):
        pass

    def generate_context_prompt(self) -> str:
        context_prompt = ("Context - Here is some information about important entities and verbs in the statement:")
        return context_prompt

    def get_prompt(self) -> List[str]:
        return [
            self.generate_context_prompt()
        ]