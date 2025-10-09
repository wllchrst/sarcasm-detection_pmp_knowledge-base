# Context-Augmented Sarcasm Detection for LLMs

An experimental framework to enhance LLM-based sarcasm detection by augmenting Pragmatic Metacognitive Prompting (PMP) with external knowledge from Named Entity Recognition (NER) and Retrieval-Augmented Generation (RAG).

**Project Status:** 🚧 On-going Research

---

## 📑 Table of Contents

- [Introduction](#-introduction)
- [Core Methodology](#-core-methodology)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#1-prerequisites)
  - [Installation](#2-installation)
  - [Setup](#3-setup)
- [Running Experiments](#-running-experiments)
  - [Example Command](#example-command)
  - [Command-Line Arguments](#command-line-arguments)
- [Understanding the Output](#-understanding-the-output)
- [Authors & Acknowledgments](#-authors--acknowledgments)
- [License](#-license)

---

## 📖 Introduction

Standard Large Language Models (LLMs) often struggle with the nuances of sarcasm, as it heavily relies on real-world context, shared knowledge, and implicit cues that are not always present in the text itself. While advanced prompting techniques like Pragmatic Metacognitive Prompting (PMP) have shown significant promise by structuring the model's reasoning process, they can still be limited by the knowledge contained within the model's parameters.

This project explores methods to systematically inject external, contextual knowledge into the prompting pipeline. We hypothesize that by identifying key entities within a sentence and providing the LLM with relevant information about them, we can further improve its ability to accurately detect sarcasm.

## 💡 Core Methodology

Our pipeline follows a modular, multi-stage process to analyze a given text for sarcasm. The core flow is as follows:

1.  **Input Processing:** A sentence from a dataset (e.g., SemEval, MUStARD) is taken as input.
2.  **Contextual Signal Extraction (Optional):**
    * **NER:** If `--use_ner` is enabled, `spaCy` is used to identify proper nouns and key entities in the sentence.
    * **RAG:** For each entity found, external knowledge is retrieved.
        * By default, the same LLM is queried for a brief description of the entity.
        * If `--use_wiki` is enabled, a local Wikipedia index is queried via **Elasticsearch (BM25)** for more robust information.
    * **Feature Engineering:** Additional signals, like verb information (`--use_verb_info`), can also be extracted.
3.  **Augmented Prompting:** The retrieved contextual information is dynamically inserted into the PMP prompt template.
4.  **LLM Inference:** The final, context-rich prompt is sent to an LLM (e.g., `qwen3:8b` via Ollama) for sarcasm detection.
5.  **Evaluation:** The model's prediction is compared against the ground truth, and performance is measured using multiple metrics.

## ✨ Key Features

* **Modular Framework:** Easily swap datasets, LLMs, and prompting techniques via command-line arguments.
* **Pragmatic Metacognitive Prompting (PMP):** Implements the advanced, multi-step PMP strategy as the baseline.
* **Contextual Augmentation:** Enhances prompts with external knowledge using:
    * Named Entity Recognition (`spaCy`)
    * Retrieval-Augmented Generation (RAG) from a local Wikipedia dump via `Elasticsearch`.
    * LLM-based entity description as an alternative RAG source.
* **Comprehensive Evaluation:** Automatically calculates Accuracy, Precision, Recall, F1-score, and generates a confusion matrix with a visual plot.
* **Reproducibility:** All results are saved in timestamped, clearly named folders for easy tracking of experiments.

## 🛠️ Tech Stack

* **Core:** Python 3.10+
* **LLM Serving:** Ollama
* **NLP & NER:** spaCy
* **RAG/Search:** Elasticsearch
* **Key Libraries:** `transformers`, `pandas`, `scikit-learn`

---

## 🚀 Getting Started

Follow these instructions to set up the environment and run experiments.

### 1. Prerequisites

* **Python:** A modern version of Python is required (developed on 3.10).
* **Ollama:** You must have Ollama installed and running. Follow the official guide at [ollama.com](https://ollama.com/).
* **Elasticsearch:** For using the `--use_wiki` feature, a local instance of Elasticsearch must be running. Follow the official guide at [elastic.co](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html). You will also need to have indexed a Wikipedia data dump.

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/wllchrst/sarcasm-detection_pmp_knowledge-base.git
    ```

2.  **Install Python dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

### 3. Setup

1.  **Pull the LLM model:**
    Make sure your Ollama service is running, then pull the model used in the experiments.
    ```bash
    ollama pull qwen3:8b
    ```

2.  **Download spaCy model:**
    You'll need the English model for NER.
    ```bash
    python -m spacy download en_core_web_sm
    ```

## 🔬 Running Experiments

All experiments are run from the root directory using the `main` module. The behavior is controlled via command-line arguments.

### Example Command

Here is an example command that runs an experiment using the SemEval dataset, the PMP prompt, and all contextual augmentation features:

```bash
python -m main --dataset semeval --llm_model qwen3:8b --prompt pmp --use_ner --use_wiki --use_verb_info --folder_name my_first_experiment
```

### Command-Line Arguments

| Argument            | Description                                                                    | Example                    |
| ------------------- | ------------------------------------------------------------------------------ | -------------------------- |
| `--dataset`         | The dataset to use for evaluation.                                             | `semeval`, `mustard`       |
| `--llm_model`       | The Ollama model to use for inference.                                         | `qwen3:8b`, `llama3:8b`    |
| `--prompt`          | The prompting strategy to use.                                                 | `pmp`                      |
| `--use_ner`         | **Flag:** Enable Named Entity Recognition to extract context.                  | `--use_ner`                |
| `--use_wiki`        | **Flag:** Use Elasticsearch/Wikipedia for RAG. If disabled, uses LLM for info. | `--use_wiki`               |
| `--use_verb_info`   | **Flag:** Include verb-based features in the context.                          | `--use_verb_info`          |
| `--sentiment_model` | The sentiment model to use for certain prompt strategies.                      | `bert_tweet`               |
| `--folder_name`     | A custom name for the output folder where results will be saved.               | `final_run_with_rag`       |

## 📈 Understanding the Output

The script will create a new directory inside the `evaluation_result/` folder. The name will be a combination of the model, dataset, and techniques used, plus your custom `--folder_name`. Inside this directory, you will find:

* `confusion_matrix.json`: A JSON file containing the model's predictions, ground truth, and evaluation metrics (Accuracy, Precision, Recall, F1).
* `confusion_matrix.png`: A visual plot of the confusion matrix.
* `log.txt`: A log of the experiment run.

---

## 🎓 Authors & Acknowledgments

This research is being conducted as a collaborative project by:
* William Christian
* Michael Iskandardinata
* **Prof. Derwin Suhartono** (NLP Specialist, Bina Nusantara University)

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
