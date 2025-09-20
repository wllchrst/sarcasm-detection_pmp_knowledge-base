import spacy
from helpers import WordHelper, env_helper
from typing import Tuple
from ner.ner_processor import NERProcessor
from interfaces import LLMType

SENTIMENT_LABEL_MAPPING_DESCRIPTION = {
    'NEG': 'Negative meaning',
    'NEU': 'Neutral meaning',
    'POS': 'Positive meaning'
}

class NEREntry:
    def __init__(self):
        self.spacy_model = spacy.load(env_helper.SPACY_MODEL)
        self.processor = NERProcessor(LLMType.OLLAMA, model_name="qwen3:8b")

    def get_sentence_token(self, text: str) -> Tuple[list, list, list]:
        doc = self.spacy_model(text)

        verbs = [token.text for token in doc if token.pos_ == "VERB"]
        entities = [entity.text for entity in doc.ents]

        return verbs, entities

    def get_sentence_context(self, sentence: str) -> str:
        cleaned = WordHelper.clean_sentence(sentence)
        verbs, entities = self.get_sentence_token(cleaned)

        verb_sentiments = self.processor.process_verbs(verbs, cleaned)
        entities_informations = self.processor.get_word_information(entities)
        verbs_informations = self.processor.get_word_information(verbs)

        verb_conclusion = ''
        for verb, sentiment in zip(verbs, verb_sentiments):
            sentiment_label = SENTIMENT_LABEL_MAPPING_DESCRIPTION\
                .get(sentiment['label'], 'Unknown sentiment')
            
            verb_conclusion += f'Verb: {verb}, Sentiment: {sentiment_label}\n'
        
        verb_info_conclusion = ''
        for verb, information in zip(verbs, verbs_informations):
            verb_info_conclusion += f'Verb: {verb}\nInformation: {information}\n'
            
        entity_conclusion = ''
        for entity, information in zip(entities, entities_informations):
            entity_conclusion += f'Entity: {entity}\nInformation: {information}\n'

        final_result = f'\n{verb_conclusion}\n{verb_info_conclusion}\n{entity_conclusion}\n'
        return final_result
