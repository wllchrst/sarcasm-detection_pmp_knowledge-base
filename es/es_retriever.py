import pandas as pd
import traceback
from typing import Dict, List
from elasticsearch import Elasticsearch
from helpers import env_helper
from interfaces import WikipediaData
from es.index_builder import IndexBuilder


def convert_hits_to_wiki(hits: List) -> List[WikipediaData]:
    data: List[WikipediaData] = []
    for hit in hits:
        source = hit['_source']
        wiki_data: WikipediaData = {
            'title': source['title'],
            'text': source['text'],
            'aliases': source['aliases']
        }

        data.append(wiki_data)

    return data


class EsRetriever:
    def __init__(self):
        self.es = Elasticsearch(env_helper.ELASTIC_HOST)
        self.index_builder = IndexBuilder(self.es)
        print(f'ES Server info: {self.es.info()}')

    def search_wiki_data(self,
                         index: str,
                         query: str,
                         total_result: int = 1) -> List[WikipediaData]:
        if not self.es.indices.exists(index=index):
            raise ValueError(f'Elastic search index {index} does not exist')

        query_result = self.es.search(
            index=index,
            size=total_result,
            query={
                'match': {
                    'title': query
                }
            }
        )
        hits = query_result['hits']['hits']

        return convert_hits_to_wiki(hits)

    def test_index(self, index: str):
        res = self.es.search(
            index=index,
            query={
                "match_all": {}
            }
        )

        print(res)
