from llm.ollama_llm import OllamaLLM
from prompt import BasePromptHandler

class PMPHandler(BasePromptHandler):
    def __init__(self):
        self.llm_model = OllamaLLM()

    def generate_initial_prompt(self) -> str:
        initial_prompt = ("You will be given a tweet from twitter, and will analyze the statement. Repeat back the statement to analyze."
                  "Then, analyze the following:\n"
                  "-What does the speaker imply about the situation with their statement?\n"
                  "-What does the speaker think about the situation?\n"
                  "-Are what the speaker implies and what the speaker thinks saying the same thing?\n"
                  "Finally, decide if the speaker is pretending to have a certain attitude toward the conversation."
        )
        return initial_prompt

    def generate_reflection_prompt(self) -> str:
        reflection_prompt = ("You will be given a statement from twitter and a preliminary analysis on the statement. Summarize the preliminary analysis"
                             "Decide whether statement is sarcastic or not by first analyzing the following:\n"
                             "\nThe Implicature - What is implied in the conversation beyond the literal meaning?"
                             "\nThe Presuppositions - What information in the conversation is taken for granted?"
                             "\nThe intent of the speaker - What do the speaker(s) hope to achieve with their statement and who are the speakers?\n"
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

    def process_response(self, response: str) -> int:
        if "yes" in response.lower():
            return 1
        else:
            return 0

    def get_response(self, text: str) -> int:
        judge_input = ""

        initial_prompt = self.generate_initial_prompt()
        initial_response = self.llm_model.answer(initial_prompt, text)

        print(initial_response)
        judge_input += "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - \n"
        judge_input += initial_response.strip()
        print(judge_input)

        reflection_prompt = self.generate_reflection_prompt()
        reflection_response = self.llm_model.answer(reflection_prompt, text + " " + judge_input)

        judge_input += "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - \n"
        judge_input += reflection_response.strip()
        judge_input += "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - \n"

        final_decision_prompt = self.generate_final_decision_prompt()
        final_response = self.llm_model.answer(final_decision_prompt, judge_input)
        print("final_response", final_response)

        return self.process_response(final_response)
