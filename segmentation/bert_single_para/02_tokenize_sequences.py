import pandas as pd
import torch
from transformers import BertTokenizer
import numpy as np
import os

import pickle

if __name__ == '__main__':
    # Use appropriate locations
    training_data_loc = 'train_sequences.csv'
    output_loc = 'train_sequences_tokenized.csv'
    
    print('Reading training data file...')
    
    df = pd.read_csv(training_data_loc)
    
    print('Loading tokenizer...')
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    
    print('Tokenizing para1...')
    df['para1_tokens'] = df['para1'].apply(lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(x))))
    df['para1_len'] = df['para1_tokens'].apply(lambda x: len(x))
    
    print('Tokenizing para2...')
    df['para2_tokens'] = df['para2'].apply(lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(x))))
    df['para2_len'] = df['para2_tokens'].apply(lambda x: len(x))
    
    
    df.to_csv(output_loc)
