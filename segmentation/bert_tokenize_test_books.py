import pandas as pd
import lxml.etree as ET
import random
import os

import pickle
import multiprocessing
from functools import partial

from transformers import BertTokenizer

def process_book(sent_dir, bert_tok_dir, tokenizer, book_id):
    try:
        filename = os.path.join(sent_dir, book_id + '.xml')
        parser = ET.XMLParser(huge_tree=True)
        tree = ET.parse(filename, parser=parser)
        book = tree.getroot()
        b = book.find('.//body')

        paragraphs = b.findall('.//p')

        d = dict()
        for p in paragraphs:
            n = int(p.attrib['num'])
            text = ' '.join([x.text for x in p.findall('.//s')])

            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(text)))

            d[n] = tokens

        with open(os.path.join(bert_tok_dir, book_id + '.pkl'), 'wb') as f:
            pickle.dump(d, f)

        return book_id, 0
    except:
        return book_id, -1
    

if __name__ == '__main__':
    # Use appropriate locations
    test_books_list_file = 'test_books.txt'
    
    sent_dir = 'use_books_sentencized/'
    bert_tok_dir = 'test_books_bert_tok/'
    
    with open(test_books_list_file, 'r') as f:
        test_book_ids = [x.strip() for x in f.readlines()]

    print(len(test_book_ids), 'books')
    
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    
    
    
    func = partial(process_book, sent_dir, bert_tok_dir, tokenizer)
    
    pool = multiprocessing.Pool(processes=32)
    data = pool.map(func, test_book_ids)
    pool.close()
    pool.join()
    
    df = pd.DataFrame(data)
    df.rename(columns={0:'book_id', 1:'status'}, inplace=True)
    df.to_csv('log_test_tok_bert.csv', index=False)
    print('Done!')

