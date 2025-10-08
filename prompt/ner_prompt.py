from prompt import BasePrompt
from typing import Dict


class NERPrompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def generate_context_prompt(self, is_indonesian: bool) -> str:
        return (
            "Konteks - Berikut adalah beberapa informasi tentang entitas dan kata kerja penting dalam pernyataan:"
            if is_indonesian
            else "Context - Here is some information about important entities and verbs in the statement:"
        )

    def get_prompt(self, is_indonesian: bool = False) -> Dict[str, str]:
        """Return a dictionary containing all prompts in English or Indonesian."""
        return {
            "context_prompt": self.generate_context_prompt(is_indonesian),
        }
