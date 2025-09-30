import argparse

parser = argparse.ArgumentParser(description="Python script that is used for indo adaptive rag experiments")


def parse_all_args():
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
    parser.add_argument("--with_logging", help="Is the evaluation going to log it into terminal", action='store_true')
    parser.add_argument("--folder_name", help="Name appended at the end of the default folder name for evaluation")
    parser.add_argument("--is_indonesian", help="Prompt that is contructed using indonesian language",
                        action='store_true')
    parser.add_argument("--context_full_llm", help="Context using full llm (from ner to information)",
                        action='store_true')
    return parser.parse_args()


def main():
    from evaluation_system import System
    from interfaces import SystemArgument
    arguments = parse_all_args()

    system_argument = SystemArgument(
        dataset=arguments.dataset,
        use_context=arguments.use_context,
        with_logging=arguments.with_logging,
        llm_model=arguments.llm_model,
        prompt=arguments.prompt,
        use_ner=arguments.use_ner,
        sentiment_model=arguments.sentiment_model,
        use_wiki=arguments.use_wiki,
        use_verb_info=arguments.use_verb_info,
        folder_name=arguments.folder_name,
        is_indonesian=arguments.is_indonesian,
        context_full_llm=arguments.context_full_llm
    )

    system = System(system_argument)
    result = system.evaluate()
    print(result)


def populate_elastic_search():
    from es import IndexBuilder
    IndexBuilder(es=None)


if __name__ == "__main__":
    main()
