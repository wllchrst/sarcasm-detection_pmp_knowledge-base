import pandas as pd
import json
import os
import seaborn as sns
import time
from evaluation_system.dataset import load_semeval_dataset
from datetime import datetime
from interfaces import SystemArgument
from prompt import PromptHandler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt

evaluation_result_folder = 'evaluation_result'
checkpoint_filename = 'checkpoint.csv'


class System:
    def __init__(self, argument: SystemArgument):
        self.argument = argument
        self.dataset = self.load_dataset()
        self.prompt_handler = self.load_prompt_handler()
        self.argument.sentiment_model = self.argument.sentiment_model if self.argument.sentiment_model != None else ""

    def load_dataset(self) -> pd.DataFrame:
        if self.argument.dataset == "semeval":
            return load_semeval_dataset()
        else:
            raise ValueError(f"Unknown dataset: {self.argument.dataset}")

    def load_prompt_handler(self) -> PromptHandler:
        prompt = self.argument.prompt
        use_ner = getattr(self.argument, "use_ner", False)
        llm_model = self.argument.llm_model
        with_logging = self.argument.with_logging
        use_wiki = getattr(self.argument, "use_wiki", False)
        use_verb_info = getattr(self.argument, "use_verb_info", False)

        if self.argument.prompt is not None:
            return PromptHandler(prompt_method=prompt,
                                 llm_model=llm_model,
                                 sentiment_model=self.argument.sentiment_model,
                                 use_ner=use_ner,
                                 use_wiki=use_wiki,
                                 with_logging=with_logging,
                                 use_verb_info=use_verb_info)
        else:
            raise ValueError(f"Unknown prompt: {prompt}")

    def evaluate(self) -> dict:
        output_folder = self.generate_evaluation_foldername()
        print(f'Output Folder: {output_folder}')
        true_labels, predictions, times = self.evaluate_dataset(
            dataset=self.dataset,
            output_folder=output_folder
        )

        print("Successful time elapsed:", sum(times))

        ########################################  GET EVALUATION  ##########################################################
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        cm = confusion_matrix(true_labels, predictions)

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist()
        }

        output_file_evaluation = os.path.join(output_folder, "evaluation.json")
        output_file_confusion_matrix = os.path.join(output_folder, "confusion_matrix.jpg")

        ########################################  SAVE EVALUATION  ##########################################################
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

    def evaluate_dataset(self,
                         dataset: pd.DataFrame,
                         output_folder: str,
                         ):
        ids = []
        texts = []
        predictions = []
        true_labels = []
        times = []  # ⏱️ new: store inference times

        checkpoint_file_path = os.path.join(output_folder, checkpoint_filename)
        predicted_ids = []
        checkpoint_dataset = None

        if os.path.exists(checkpoint_file_path):
            checkpoint_dataset = pd.read_csv(checkpoint_file_path)
            predicted_ids = checkpoint_dataset['id'].values

        try:
            for index, row in dataset.iterrows():
                id = row['id']
                text = row['text']
                label = row['label']

                if id in predicted_ids:
                    print(f'Skipped index {index} with dataset id: {id}')
                    continue

                # ⏱️ measure classification time
                start_time = time.perf_counter()
                classification_result = self.prompt_handler.process(text, self.argument.with_logging)
                elapsed_time = time.perf_counter() - start_time

                ids.append(id)
                texts.append(text)
                predictions.append(classification_result)
                true_labels.append(label)
                times.append(elapsed_time)

                if getattr(self.argument, "with_logging", False):
                    print(f"Evaluating row {index}: {text} with label {label}")
                    print(index, "classification_result", classification_result)
                    print(f"Time taken: {elapsed_time:.4f} seconds")
                    print(
                        "\n\n---------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n"
                    )
                else:
                    print(index)

        except Exception as e:
            new_checkpoint_dataset = pd.DataFrame({
                "id": ids,
                "text": texts,
                "prediction": predictions,
                "true_label": true_labels,
                "time_taken": times,  # ⏱️ save timing
            })

            if checkpoint_dataset is not None:
                checkpoint_dataset = pd.concat([new_checkpoint_dataset, checkpoint_dataset])
            else:
                checkpoint_dataset = new_checkpoint_dataset

            checkpoint_dataset = checkpoint_dataset[["id", "text", "prediction", "true_label", "time_taken"]]
            checkpoint_dataset.to_csv(checkpoint_file_path, index=False)

            print(checkpoint_dataset.head())
            print(f"Error evaluation, checkpoint dataset saved: {checkpoint_file_path}")
            raise e

        if checkpoint_dataset is not None:
            true_labels = list(true_labels) + list(checkpoint_dataset["true_label"].values)
            predictions = list(predictions) + list(checkpoint_dataset['prediction'].values)
            times = list(times) + list(checkpoint_dataset['time_taken'].values)

        return true_labels, predictions, times

    def generate_evaluation_foldername(self) -> str:
        os.makedirs(evaluation_result_folder, exist_ok=True)

        # sanitize model names
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        llm_model = self.argument.llm_model.replace("/", "_").replace(":", "_")
        sentiment_model = self.argument.sentiment_model.replace("/", "_").replace(":", "_")
        ner_information = ""

        if self.argument.use_ner:
            if self.argument.use_wiki and self.argument.use_verb_info:
                ner_information = "ner_wiki_verb"
            elif self.argument.use_wiki:
                ner_information = "ner_wiki"
            elif self.argument.use_verb_info:
                ner_information = "ner_verb"
            else:
                ner_information = "ner"
        else:
            ner_information = ""

        foldername = f"{llm_model}_{sentiment_model}_{ner_information}{self.argument.dataset}_results"
        foldername = os.path.join(evaluation_result_folder, foldername)
        foldername += f"_{timestamp_str}"

        os.makedirs(foldername, exist_ok=True)
        return foldername
