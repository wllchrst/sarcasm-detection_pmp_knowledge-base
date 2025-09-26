from prompt import BasePrompt
from typing import Dict

class PMPPrompt(BasePrompt):
    def __init__(self, dataset: str):
        self.dataset = dataset
        super().__init__()

    def generate_initial_first_prompt(self) -> str:
        if self.dataset == "semeval":
            initial_first_prompt = ("You will be given a tweet from twitter, and will analyze the statement. Repeat back the statement to analyze. ")
        elif self.dataset == "mustard":
            initial_first_prompt = ("You will be given movie or tv show dialogue, and will analyze the statement marked between brackets. Summarize the conversation, and repeat back the statement to analyze. ")
        else:
            initial_first_prompt = ("You will be given a text, and will analyze the statement. Repeat back the statement to analyze. ")
        return initial_first_prompt

    def generate_initial_prompt(self) -> str:
        initial_prompt = ("Then, analyze the following:\n"
                          "-What does the speaker imply about the situation with their statement?\n"
                          "-What does the speaker think about the situation?\n"
                          "-Are what the speaker implies and what the speaker thinks saying the same thing?\n"
        )
        return initial_prompt

    def generate_initial_last_prompt(self) -> str:
        initial_last_prompt = "Finally, decide if the speaker is pretending to have a certain attitude toward the conversation."
        return initial_last_prompt
    
    def generate_reflection_first_prompt(self) -> str:
        if self.dataset == "semeval":
            reflection_first_prompt = ("You will be given a statement from twitter and a preliminary analysis on the statement. Summarize the preliminary analysis. ")
        elif self.dataset == "mustard":
            reflection_first_prompt = ("You will be given a piece of movie dialogue, a statement marked in brackets, and a preliminary analysis on the marked statement. Summarize the preliminary analysis. ")
        else:
            reflection_first_prompt = ("You will be given a statement and a preliminary analysis on the statement. Summarize the preliminary analysis. ")
        return reflection_first_prompt

    def generate_reflection_prompt(self) -> str:
        reflection_prompt = ("Decide whether statement is sarcastic or not by first analyzing the following:"
                             "\nThe Implicature - What is implied in the conversation beyond the literal meaning?"
                             "\nThe Presuppositions - What information in the conversation is taken for granted?"
                             "\nThe intent of the speaker - What do the speaker(s) hope to achieve with their statement and who are the speakers?"
                             "\nThe polarity - Does the last sentence have a positive or negative tone?"
                             "\nPretense - Is there pretense in the speaker's attitude?"
                             "\nMeaning- What is the difference between the literal and implied meaning of the statement?"
                              "Reflect on the preliminary analysis and what should change, then decide if the statment is sarcastic.")
        return reflection_prompt

    def generate_final_decision_prompt(self) -> str:
        final_decision_prompt = (
            "You will be given the output of an LLM which decided if a sentence is sarcastic or not. "
            "Read the output, then summarize the LLM's stance with ONLY a YES (they think the sentence is sarcastic) or NO (they think the sentence is not sarcastic)."
        )
        return final_decision_prompt

    def get_prompt(self) -> Dict[str, str]:
        return {
            "initial_first_prompt": self.generate_initial_first_prompt(),
            "initial_prompt": self.generate_initial_prompt(),
            "initial_last_prompt": self.generate_initial_last_prompt(),
            "reflection_first_prompt": self.generate_reflection_first_prompt(),
            "reflection_prompt": self.generate_reflection_prompt(),
            "final_decision_prompt": self.generate_final_decision_prompt(),
        }