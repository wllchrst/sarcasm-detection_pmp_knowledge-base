import argparse


class ArgumentHelper:
    @staticmethod
    def parse_main_script():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", help="Dataset that is going to evaluated")
        parser.add_argument("--llm_model", help="LLM that is going to be used for ollama")
        parser.add_argument("--prompt", help="Prompting technique that is going to be used")
        parser.add_argument("--use_ner", help="Is the prompting technique going to use ner information",
                            action='store_true')
        parser.add_argument("--use_wiki", help="Verb information from wiki or llm",
                            action='store_true')
        parser.add_argument("--use_verb_info", help="Information of verb is going to be retrieved.",
                            action='store_true')
        parser.add_argument("--sentiment_model", help="Sentiment model")
        parser.add_argument("--use_context", help="Is the evaluation going to be run using the context of the dataset",
                            action='store_true')
        parser.add_argument("--with_logging", help="Is the evaluation going to log it into terminal",
                            action='store_true')
        parser.add_argument("--folder_name", help="Name appended at the end of the default folder name for evaluation")
        parser.add_argument("--is_indonesian", help="Prompt that is contructed using indonesian language",
                            action='store_true')
        parser.add_argument("--context_full_llm", help="Context using full llm (from ner to information)",
                            action='store_true')
        return parser.parse_args()

    @staticmethod
    def parse_context_generation():
        parser = argparse.ArgumentParser()
        parser.add_argument("--partition", help="Dataset partition that context is going to be build on")

        return parser.parse_args()
