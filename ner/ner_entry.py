import spacy
from helpers import WordHelper, env_helper
from typing import Tuple
from ner.ner_processor import NERProcessor
from interfaces import LLMType


class NEREntry:
    def __init__(self):
        self.spacy_model = spacy.load(env_helper.SPACY_MODEL)
        self.processor = NERProcessor(LLMType.GEMINI)

    def get_sentence_token(self, text: str) -> Tuple[list, list, list]:
        doc = self.spacy_model(text)

        verbs = [token.text for token in doc if token.pos_ == "VERB"]
        entities = [entity.text for entity in doc.ents]

        return verbs, entities

    def get_sentence_context(self, sentence: str) -> str:
        cleaned = WordHelper.clean_sentence(sentence)
        verbs, entities = self.get_sentence_token(cleaned)

        verb_sentiments = self.processor.process_verbs(verbs, cleaned)
        entities_informations = self.processor.get_entity_information(entities)
        return ""
