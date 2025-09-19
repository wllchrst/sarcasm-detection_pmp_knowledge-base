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
        print("initial_prompt", initial_prompt)
        initial_response = self.ollama.answer(initial_prompt, text)
        print("initial_response", initial_response)

        judge_input += line_seperator
        # judge_input += initial_response.strip()
        judge_input += "\nTHIS IS INITIAL RESPONSE EXAMPLE AAAA\n"
        print("judge_input init")
        print(judge_input)
        print("judge_input init ENDDD")

        reflection_prompt = prompts[1]
        print("reflection_prompt", reflection_prompt)
        reflection_response = self.ollama.answer(reflection_prompt, text + " " + judge_input)
        print("REFLECTION input start")
        print(text + " " + judge_input)
        print("REFLECTION input end")

        judge_input += line_seperator
        # judge_input += reflection_response.strip()
        judge_input += "\nTHIS IS REFLECT RESPONSE EXAMPLE BBBBBBBBBBBBBB\n"
        judge_input += line_seperator
        print("judge_input FINALLLLL")
        print(judge_input)
        print("judge_input FINALLLLL ENDDD")

        final_decision_prompt = prompts[2]
        final_response = self.ollama.answer(final_decision_prompt, judge_input)
        print("final_response", final_response)

        return self.process_response(final_response)

    def process_response(self, response: str) -> int:
        if "yes" in response.lower():
            return 1
        else:
            return 0