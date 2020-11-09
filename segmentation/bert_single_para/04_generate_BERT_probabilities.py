import os
import pickle
import pandas as pd

import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer

import sys

def process_book(bert_tok_dir, pred_scores_dir, BertNSP, device, cls, sep, book_id):
    with open(os.path.join(bert_tok_dir, book_id + '.pkl'), 'rb') as f:
        d = pickle.load(f)
    
    m = max(d.keys())
    
    scores = dict()
    for idx in range(0, m - 1):
        toks1 = d[idx]
        toks2 = d[idx + 1]

        l1 = len(toks1)
        l2 = len(toks2)
        if l1 + l2 >= 297:
            if l1 > 148 and l2 > 148:
                toks1 = toks1[-148:]
                toks2 = toks2[:148]
            elif l1 > 148:
                rem_len = 297 - l2
                toks1 = toks1[-rem_len:]
            elif l2 > 148:
                rem_len = 297 - l1
                toks2 = toks2[:rem_len]

        ids1 = [cls] + toks1 + [sep]
        ids2 = toks2 + [sep]

        indexed_tokens = ids1 + ids2
        segments_ids = [0] * len(ids1) + [1] * len(ids2)

        indexed_tokens = pad_sequences([indexed_tokens], maxlen=300, dtype='long', value=0, truncating="pre", padding="post")
        segments_ids = pad_sequences([segments_ids], maxlen=300, dtype="long", value=1, truncating="pre", padding="post")
        attention_masks = [[int(token_id > 0) for token_id in sent] for sent in indexed_tokens]

        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)
        attention_tensor = torch.tensor(attention_masks)

        tokens_tensor = tokens_tensor.to(device)
        segments_tensors = segments_tensors.to(device)
        attention_tensor = attention_tensor.to(device)

        BertNSP.eval()
        prediction = BertNSP(tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_tensor)
        prediction = prediction[0] # tuple to tensor
        softmax = torch.nn.Softmax(dim=1)
        prediction_sm = softmax(prediction)

        scores[idx] = prediction_sm[0][1].item()
        
    with open(os.path.join(pred_scores_dir, book_id + '.pkl'), 'wb') as f:
        pickle.dump(scores, f)
    
    return


if __name__ == '__main__':
    # Use appropriate locations
    test_books_list_file = 'test_books.txt'
    
    bert_tok_dir = 'test_books_bert_tok/'
    pred_scores_dir = 'test_preds/'

    model_dir = 'model_3/'

    with open(test_books_list_file, 'r') as f:
        test_book_ids = [x.strip() for x in f.readlines()]

    partition = int(sys.argv[1])
    from_idx = partition * 1000
    to_idx = (partition + 1) * 1000

    test_book_ids = test_book_ids[from_idx:to_idx]
    print(len(test_book_ids), 'books')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    cls, sep = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
    
    
    device = torch.device("cuda:" + str(partition + 1) if torch.cuda.is_available() else "cpu")
    print(device)
    
    print(torch.cuda.device_count(), "GPUs")

    model = BertForNextSentencePrediction.from_pretrained(model_dir)
    model = model.to(device)
    
    
    for book_id in test_book_ids:
        print(book_id)
        try:
            process_book(bert_tok_dir, pred_scores_dir, model, device, cls, sep, book_id)
        except Exception as e:
            print(book_id, e)
    
    print('Done!')
