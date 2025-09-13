"""
Make sure all dataset that is loaded from this file is already processed by labeling the column name

Label Column Name: 'label'
Text Column Name: 'text'
Context Column Name: 'context'
"""
import pandas as pd

def load_semeval_dataset(file_path: str = 'SemEval2018-T3_gold_test_taskA_emoji.txt') -> pd.DataFrame:
    semeval_df = pd.read_csv(file_path, sep='\t')
    
    return semeval_df.rename(columns={'Tweet text': 'text', 'Label': 'label'})