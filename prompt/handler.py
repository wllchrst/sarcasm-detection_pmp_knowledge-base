from ner import NEREntry
from llm import OllamaLLM
from prompt import NERPrompt


class PromptHandler:
    def __init__(self,
                 prompt_method: str,
                 llm_model: str,
                 sentiment_model: str,
                 use_ner: bool = False,
                 use_wiki: bool = False,
                 use_verb_info: bool = False
                 ):
        self.prompt_method = prompt_method
        self.use_ner = use_ner
        self.use_wiki = use_wiki
        self.use_verb_info = use_verb_info
        self.ollama = OllamaLLM(llm_model)
        self.ner_entry = NEREntry(model_name=llm_model,
                                  sentiment_model=sentiment_model,
                                  use_wiki=use_wiki
                                  )

    def process(self,
                text: str,
                with_logging: bool) -> int:
        if self.prompt_method == "pmp":
            return self.pmp_process(text, with_logging)
        else:
            raise ValueError("Prompt method not set/found")

    def pmp_process(self,
                    text: str,
                    with_logging: bool) -> int:
        from prompt import PMPPrompt
        pmp_prompt = PMPPrompt()
        judge_input = ""
        line_seperator = "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - \n"

        prompts = pmp_prompt.get_prompt()
        initial_prompt = f'{prompts[0]}{prompts[1]}'

        if self.use_ner:
            ner_information = self.ner_entry.get_sentence_context(text, self.use_wiki, self.use_verb_info)

            if len(ner_information.strip()) > 0:
                ner_prompt = NERPrompt()
                context_prompt = ner_prompt.get_prompt()[0]
                initial_prompt = f'{prompts[0]}{context_prompt}{ner_information}{prompts[1]}'

        initial_response = self.ollama.answer(initial_prompt, text, with_logging)

        judge_input += line_seperator
        judge_input += initial_response.strip()

        reflection_prompt = prompts[2]
        reflection_response = self.ollama.answer(reflection_prompt, text + " " + judge_input, with_logging)

        judge_input += line_seperator
        judge_input += reflection_response.strip()
        judge_input += line_seperator

        final_decision_prompt = prompts[3]
        final_response = self.ollama.answer(final_decision_prompt, judge_input, with_logging)

        return self.process_response(final_response)

    def process_response(self, response: str) -> int:
        if "yes" in response.lower():
            return 1
        else:
            return 0
