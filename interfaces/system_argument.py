from dataclasses import dataclass

@dataclass
class SystemArgument:
    dataset: str
    use_context: bool
    with_logging: bool
    llm_model: str
    sentiment_model: str