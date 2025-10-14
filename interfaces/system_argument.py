from dataclasses import dataclass


@dataclass
class SystemArgument:
    dataset: str
    use_context: bool
    with_logging: bool
    llm_model: str
    prompt: str
    use_ner: bool
    use_wiki: bool
    use_verb_info: bool
    use_few_shot: bool
    sentiment_model: str
    folder_name: str
    is_indonesian: bool
    context_full_llm: bool
