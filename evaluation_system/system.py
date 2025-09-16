import pandas as pd
import json
import os
import seaborn as sns
from evaluation_system.dataset import load_semeval_dataset
from interfaces import SystemArgument
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt

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

            # TODO: replace with actual pipeline prediction
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

        output_folder = self.generate_evaluation_foldername()
        output_file_evaluation = os.path.join(output_folder, "evaluation.json")
        output_file_confusion_matrix = os.path.join(output_folder, "confusion_matrix.jpg")

        # Save evaluation results as JSON
        with open(output_file_evaluation, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        # Save confusion matrix as image
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_file_confusion_matrix)
        plt.close()

        return results

    def generate_evaluation_foldername(self) -> str:
        os.makedirs(evaluation_result_folder, exist_ok=True)

        # sanitize model names
        llm_model = self.argument.llm_model.replace("/", "_").replace(":", "_")
        sentiment_model = self.argument.sentiment_model.replace("/", "_").replace(":", "_")

        foldername = f"{llm_model}_{sentiment_model}_{self.argument.dataset}_results"
        foldername = os.path.join(evaluation_result_folder, foldername)

        os.makedirs(foldername, exist_ok=True)
        return foldername
