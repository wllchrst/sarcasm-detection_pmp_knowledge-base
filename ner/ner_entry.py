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
    def __init__(self, model_name: str, sentiment_model: str, use_wiki: bool, with_logging: bool):
        self.spacy_model = spacy.load(env_helper.SPACY_MODEL)
        self.processor = NERProcessor(llm_type=LLMType.OLLAMA,
                                      use_wiki=use_wiki,
                                      sentiment_model=sentiment_model,
                                      with_logging=with_logging,
                                      model_name=model_name)

    def get_sentence_token(self, text: str) -> Tuple[list, list]:
        doc = self.spacy_model(text)
        verbs = [token.text for token in doc if token.pos_ == "VERB"]
        proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
        entities = [entity.text for entity in doc.ents]

        additional_entity = []
        for chunk in doc.noun_chunks:
            for noun in proper_nouns:
                if noun in chunk.text:
                    additional_entity.append(chunk.text)

        entities = entities + additional_entity

        return verbs, entities

    def get_sentence_context(self,
                             sentence: str,
                             use_wiki: bool = False,
                             use_verb_info: bool = False) -> str:

        cleaned = WordHelper.clean_sentence(sentence)
        verbs, entities = self.get_sentence_token(cleaned)

        verb_sentiments = self.processor.process_verbs(verbs, cleaned)
        entities_information = self.processor.get_word_information(entities, use_wiki)
        verbs_information = []

        if use_verb_info:
            verbs_information = self.processor.get_word_information(verbs, use_wiki)

        verb_conclusion = ''
        for verb, sentiment in zip(verbs, verb_sentiments):
            sentiment_label = SENTIMENT_LABEL_MAPPING_DESCRIPTION \
                .get(sentiment['label'], 'Unknown sentiment')

            verb_conclusion += f'Verb: {verb}, Sentiment: {sentiment_label}\n'

        verb_info_conclusion = ''
        for verb, information in zip(verbs, verbs_information):
            verb_info_conclusion += f'Verb: {verb}\nInformation: {information}\n'

        entity_conclusion = ''
        for entity, information in zip(entities, entities_information):
            entity_conclusion += f'Entity: {entity}\nInformation: {information}\n'

        final_result = f'\n{verb_conclusion}\n{verb_info_conclusion}\n{entity_conclusion}\n'
        return final_result
