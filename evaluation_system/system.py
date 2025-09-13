import pandas as pd
import json
import os
from evaluation_system.dataset import load_semeval_dataset
from interfaces import SystemArgument
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

evaluation_result_folder = 'evaluation_result'

class System:
    def __init__(self, arguement: SystemArgument):
        self.argument = arguement
        self.dataset = self.load_dataset()

    def load_dataset(self) -> pd.DataFrame:
        if self.argument.dataset == "semeval":
            return load_semeval_dataset()
        else:
            raise ValueError(f"Unknown dataset: {self.argument.dataset}")

    def evaluate(self) -> dict:
        predictions = []
        true_labels = []

        for index, row in self.dataset.iterrows():
            text = row['text']
            label = row['label']

            if getattr(self.argument, "with_logging", False):
                print(f"Evaluating row {index}: {text} with label {label}")

            # TODO: MASUKIN SINI MICH LU PUNYA PIPELINE YANG UDA PAKE PUNYA GW NER
            classification_result = 1
            predictions.append(classification_result)
            true_labels.append(label)

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        cm = confusion_matrix(true_labels, predictions)

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist()
        }

        output_file = self.generate_evaluation_filename()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
    

    def generate_evaluation_filename(self) -> str:
        os.makedirs(evaluation_result_folder, exist_ok=True)

        # sanitize model names
        llm_model = self.argument.llm_model.replace("/", "_").replace(":", "_")
        sentiment_model = self.argument.sentiment_model.replace("/", "_").replace(":", "_")

        filename = f"{llm_model}_{sentiment_model}_{self.argument.dataset}_results.csv"
        return os.path.join(evaluation_result_folder, filename)