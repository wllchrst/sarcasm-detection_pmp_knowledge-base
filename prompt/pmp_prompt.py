from prompt import BasePrompt
from typing import Dict


class PMPPrompt(BasePrompt):
    """
    Class for generating PMP Prompts based on:
    https://github.com/wyatt-fong/Pragmatic-Metacognitive-Prompting-Improves-LLM-Performance-on-Sarcasm-Detection
    """

    def __init__(self, dataset: str):
        self.dataset = dataset
        super().__init__()

    def generate_initial_first_prompt(self, is_indonesian: bool) -> str:
        if is_indonesian:
            if self.dataset == "semeval":
                return "Kamu akan diberikan sebuah tweet dari Twitter dan diminta untuk menganalisis pernyataan tersebut. Ulangi kembali pernyataan yang akan dianalisis."
            elif self.dataset == "mustard":
                return "Kamu akan diberikan dialog dari film atau acara TV, dan diminta untuk menganalisis pernyataan yang ditandai dengan tanda kurung. Ringkas percakapannya, lalu ulangi pernyataan yang akan dianalisis."
            else:
                return "Kamu akan diberikan sebuah teks dan diminta untuk menganalisis pernyataan di dalamnya. Ulangi kembali pernyataan yang akan dianalisis."
        else:
            if self.dataset == "semeval":
                return "You will be given a tweet from Twitter, and will analyze the statement. Repeat back the statement to analyze."
            elif self.dataset == "mustard":
                return "You will be given movie or TV show dialogue, and will analyze the statement marked between brackets. Summarize the conversation, and repeat back the statement to analyze."
            else:
                return "You will be given a text, and will analyze the statement. Repeat back the statement to analyze."

    def generate_initial_prompt(self, is_indonesian: bool) -> str:
        return (
            "Kemudian, analisis hal-hal berikut:\n"
            "- Apa yang diimplikasikan oleh pembicara tentang situasi melalui pernyataannya?\n"
            "- Apa yang dipikirkan pembicara tentang situasi tersebut?\n"
            "- Apakah yang diimplikasikan dan yang dipikirkan pembicara menyampaikan hal yang sama?\n"
            if is_indonesian else
            "Then, analyze the following:\n"
            "- What does the speaker imply about the situation with their statement?\n"
            "- What does the speaker think about the situation?\n"
            "- Are what the speaker implies and what the speaker thinks saying the same thing?\n"
        )

    def generate_initial_last_prompt(self, is_indonesian: bool) -> str:
        return (
            "Terakhir, tentukan apakah pembicara berpura-pura memiliki sikap tertentu terhadap percakapan tersebut."
            if is_indonesian
            else "Finally, decide if the speaker is pretending to have a certain attitude toward the conversation."
        )

    def generate_reflection_first_prompt(self, is_indonesian: bool) -> str:
        if is_indonesian:
            if self.dataset == "semeval":
                return "Kamu akan diberikan sebuah pernyataan dari Twitter dan analisis awal terhadap pernyataan tersebut. Ringkas analisis awal tersebut."
            elif self.dataset == "mustard":
                return "Kamu akan diberikan potongan dialog film, sebuah pernyataan yang ditandai dengan tanda kurung, dan analisis awal terhadap pernyataan tersebut. Ringkas analisis awal tersebut."
            else:
                return "Kamu akan diberikan sebuah pernyataan dan analisis awal terhadap pernyataan tersebut. Ringkas analisis awal tersebut."
        else:
            if self.dataset == "semeval":
                return "You will be given a statement from Twitter and a preliminary analysis on the statement. Summarize the preliminary analysis."
            elif self.dataset == "mustard":
                return "You will be given a piece of movie dialogue, a statement marked in brackets, and a preliminary analysis on the marked statement. Summarize the preliminary analysis."
            else:
                return "You will be given a statement and a preliminary analysis on the statement. Summarize the preliminary analysis."

    def generate_reflection_prompt(self, is_indonesian: bool) -> str:
        return (
            "Tentukan apakah pernyataan tersebut bersifat sarkastik atau tidak dengan terlebih dahulu menganalisis hal-hal berikut:\n"
            "Implikatur - Apa yang tersirat dalam percakapan di luar makna literal?\n"
            "Presuposisi - Informasi apa dalam percakapan yang dianggap sudah diketahui?\n"
            "Niat pembicara - Apa yang ingin dicapai pembicara dengan pernyataannya dan siapa pembicaranya?\n"
            "Polaritas - Apakah kalimat terakhir bernada positif atau negatif?\n"
            "Kepura-puraan - Apakah ada kepura-puraan dalam sikap pembicara?\n"
            "Makna - Apa perbedaan antara makna literal dan makna tersirat dari pernyataan tersebut?\n"
            "Renungkan analisis awal dan apa yang perlu diubah, lalu tentukan apakah pernyataan tersebut bersifat sarkastik."
            if is_indonesian else
            "Decide whether the statement is sarcastic or not by first analyzing the following:\n"
            "The Implicature - What is implied in the conversation beyond the literal meaning?\n"
            "The Presuppositions - What information in the conversation is taken for granted?\n"
            "The intent of the speaker - What do the speaker(s) hope to achieve with their statement and who are the speakers?\n"
            "The polarity - Does the last sentence have a positive or negative tone?\n"
            "Pretense - Is there pretense in the speaker's attitude?\n"
            "Meaning - What is the difference between the literal and implied meaning of the statement?\n"
            "Reflect on the preliminary analysis and what should change, then decide if the statement is sarcastic."
        )

    def generate_final_decision_prompt(self, is_indonesian: bool) -> str:
        return (
            "Kamu akan diberikan keluaran dari sebuah model LLM yang telah memutuskan apakah sebuah kalimat bersifat sarkastik atau tidak. "
            "Baca hasil tersebut, lalu simpulkan pendapat LLM hanya dengan YA (model berpikir kalimat tersebut sarkastik) atau TIDAK (model berpikir kalimat tersebut tidak sarkastik)."
            if is_indonesian else
            "You will be given the output of an LLM which decided if a sentence is sarcastic or not. "
            "Read the output, then summarize the LLM's stance with ONLY a YES (they think the sentence is sarcastic) or NO (they think the sentence is not sarcastic)."
        )

    def get_prompt(self, is_indonesian: bool = False) -> Dict[str, str]:
        """Return a dictionary containing all prompts in English or Indonesian."""
        return {
            "initial_first_prompt": self.generate_initial_first_prompt(is_indonesian),
            "initial_prompt": self.generate_initial_prompt(is_indonesian),
            "initial_last_prompt": self.generate_initial_last_prompt(is_indonesian),
            "reflection_first_prompt": self.generate_reflection_first_prompt(is_indonesian),
            "reflection_prompt": self.generate_reflection_prompt(is_indonesian),
            "final_decision_prompt": self.generate_final_decision_prompt(is_indonesian),
        }
