from dataclasses import dataclass

@dataclass
class SystemArgument:
    dataset: str
    use_context: bool
    with_logging: bool
    llm_model: str
    prompt: str
    use_ner: bool
    sentiment_model: str