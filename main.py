def main():
    from evaluation_system import System
    from interfaces import SystemArgument
    from helpers.argument_helper import ArgumentHelper
    arguments = ArgumentHelper.parse_main_script()

    system_argument = SystemArgument(
        dataset=arguments.dataset,
        use_context=arguments.use_context,
        with_logging=arguments.with_logging,
        llm_model=arguments.llm_model,
        prompt=arguments.prompt,
        use_few_shot=arguments.use_few_shot,
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
