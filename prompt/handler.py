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
                 use_verb_info: bool = False,
                 with_logging: bool = False
                 ):
        self.prompt_method = prompt_method
        self.use_ner = use_ner
        self.use_wiki = use_wiki
        self.use_verb_info = use_verb_info
        self.ollama = OllamaLLM(llm_model)
        self.ner_entry = NEREntry(model_name=llm_model,
                                  sentiment_model=sentiment_model,
                                  use_wiki=use_wiki,
                                  with_logging=with_logging
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
        log_separator = "=" * 100
        
        if with_logging:
            print(log_separator)

        prompts = pmp_prompt.get_prompt()
        initial_prompt = prompts.get('initial_prompt')
        initial_last_prompt = prompts.get('initial_last_prompt')
        combined_initial_prompt = f'{initial_prompt}{initial_last_prompt}'

        if self.use_ner:
            ner_information = self.ner_entry.get_sentence_context(text, self.use_wiki, self.use_verb_info)

            if len(ner_information.strip()) > 0:
                ner_prompt = NERPrompt()
                context_prompt = ner_prompt.get_prompt().get('context_prompt')            
                combined_initial_prompt = f'{initial_prompt}{context_prompt}{ner_information}{initial_last_prompt}'

        initial_response = self.ollama.answer(combined_initial_prompt, text, with_logging)

        judge_input += line_seperator
        judge_input += initial_response.strip()

        reflection_prompt = prompts.get('reflection_prompt')
        reflection_response = self.ollama.answer(reflection_prompt, text + " " + judge_input, with_logging)

        judge_input += line_seperator
        judge_input += reflection_response.strip()
        judge_input += line_seperator

        final_decision_prompt = prompts.get('final_decision_prompt')
        final_response = self.ollama.answer(final_decision_prompt, judge_input, with_logging)

        return self.process_response(final_response)

    def process_response(self, response: str) -> int:
        if "yes" in response.lower():
            return 1
        else:
            return 0
