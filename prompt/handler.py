from prompt import PMPPrompt
from llm import OllamaLLM

class PromptHandler:
    def __init__(self, prompt_method: str, llm_model: str):
        self.prompt_method = prompt_method
        self.ollama = OllamaLLM(llm_model)

    def process(self, text: str) -> int:
        if self.prompt_method == "pmp":
            return self.pmp_process(text)
        else:
            raise ValueError("Prompt method not set/found")

    def pmp_process(self, text: str) -> int:
        pmp_prompt = PMPPrompt()
        judge_input = ""
        line_seperator = "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - \n"
        prompts = pmp_prompt.get_prompt()

        initial_prompt = prompts[0]
        initial_response = self.ollama.answer(initial_prompt, text)

        judge_input += line_seperator
        judge_input += initial_response.strip()

        reflection_prompt = prompts[1]
        reflection_response = self.ollama.answer(reflection_prompt, text + " " + judge_input)

        judge_input += line_seperator
        judge_input += reflection_response.strip()
        judge_input += line_seperator

        final_decision_prompt = prompts[2]
        final_response = self.ollama.answer(final_decision_prompt, judge_input)

        return self.process_response(final_response)

    def process_response(self, response: str) -> int:
        if "yes" in response.lower():
            return 1
        else:
            return 0