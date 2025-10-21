from prompt import BasePrompt
from typing import Dict


class NERPrompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def generate_context_system_prompt(self, is_indonesian: bool) -> str:
        return (
            "Selain itu, ada disediakan beberapa fakta entitas dari kalimat yang dapat Anda gunakan. Hanya gunakan fakta tersebut jika langsung relevan, JANGAN menciptakan fakta baru."
            if is_indonesian
            else "There are also some entity facts from the sentence that you can use. Only use them if directly relevant, do NOT invent new facts."
        )
    
    def generate_context_user_prompt(self, is_indonesian: bool) -> str:
        return (
            "\nDefinisi kata-kata penting:\n"
            if is_indonesian
            else "Entity facts:\n"
        )

    def get_prompt(self, is_indonesian: bool = False) -> Dict[str, str]:
        """Return a dictionary containing all prompts in English or Indonesian."""
        return {
            "context_system_prompt": self.generate_context_system_prompt(is_indonesian),
            "context_user_prompt": self.generate_context_user_prompt(is_indonesian),
        }
