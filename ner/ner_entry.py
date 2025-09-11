import spacy
from helpers import WordHelper, env_helper
from typing import Tuple
from ner.ner_processor import NERProcessor
from interfaces import LLMType


class NEREntry:
    def __init__(self):
        self.spacy_model = spacy.load(env_helper.SPACY_MODEL)
        self.processor = NERProcessor(LLMType.GEMINI)

    def get_sentence_nouns(self, text: str) -> Tuple[list, list, list]:
        doc = self.spacy_model(text)

        nouns = [noun.text for noun in doc.noun_chunks]
        verbs = [token.text for token in doc if token.pos_ == "VERB"]
        entities = [entity.text for entity in doc.ents]

        return nouns, verbs, entities

    def get_sentence_context(self, sentence: str) -> str:
        cleaned = WordHelper.clean_sentence(sentence)
        nouns, verbs, entities = self.get_sentence_nouns(cleaned)

        self.processor.process_verbs(entities, cleaned)

        return ""
