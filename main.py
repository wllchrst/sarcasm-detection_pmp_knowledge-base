def main():
    from ner import NEREntry

    entry = NEREntry()
    context = entry.get_sentence_context("I think Elon Musk is really needs more money to buy food")
    print(context)

# TODOLIST
# 1. Finish NER Evaluation to give context of the sentence
# 2. Create Evaluation pipeline.
# 3. Add logging using MLFLOW
if __name__ == "__main__":
    main()