import pandas as pd
import traceback
from typing import Dict, List, Optional
from elasticsearch import Elasticsearch
from helpers import env_helper

WIKIPEDIA_DATA_INDEX = 'wikipedia'


class IndexBuilder:
    def __init__(self, es: Optional[Elasticsearch]):
        self.es = es if es is not None else Elasticsearch(env_helper.ELASTIC_HOST)
        print(self.es.info())

        self.build_necessary_index()

    def build_necessary_index(self):
        self.build_wiki_index(override=False)

    def create_insert_index(self, index_name: str, documents: List[Dict]):
        self.es.indices.create(index=index_name)
        operations = []
        for document in documents:
            operations.append({'index': {'_index': index_name}})
            operations.append(document)

        self.es.bulk(operations=operations)

    def build_wiki_index(self, override: bool) -> bool:
        exists = self.es.indices.exists(index=WIKIPEDIA_DATA_INDEX)
        if exists and not override:
            print(f"Index {WIKIPEDIA_DATA_INDEX} already exist in the dataset")
            return True
        elif exists and override:
            self.es.indices.delete(index=WIKIPEDIA_DATA_INDEX)

        try:
            documents = self.load_wiki_dataset()
            self.create_insert_index(WIKIPEDIA_DATA_INDEX, documents)

            return True
        except Exception as e:
            traceback.print_exc()
            raise e

    # Dataset installed from: https://github.com/declare-lab/WikiDes/blob/main/dataset/collected_data.zip
    def load_wiki_dataset(self, path: str = 'collected_data.json') -> List[Dict]:
        dataset = pd.read_json(path)
        documents: List[Dict] = []

        for index, row in dataset.iterrows():
            documents.append({
                'title': row['label'],
                'text': row['description'],
                'aliases': row['aliases']
            })

        return documents
