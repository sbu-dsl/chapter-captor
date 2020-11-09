import pandas as pd
import lxml.etree as ET
import random
import os

import pickle
import multiprocessing
from functools import partial

def process_book(sent_dir, para_to_sent_dir, book_id):
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
            
            sents = p.findall('.//s')
            num = int(sents[-1].attrib['num'])
            d[n] = num

        with open(os.path.join(para_to_sent_dir, book_id + '.pkl'), 'wb') as f:
            pickle.dump(d, f)

        return book_id, 0
    except:
        return book_id, -1
    

if __name__ == '__main__':
    # Use appropriate locations
    test_books_list_file = 'test_books.txt'
    
    sent_dir = 'use_books_sentencized/'
    para_to_sent_dir = 'test_books_para_to_sent/'
    
    with open(test_books_list_file, 'r') as f:
        test_book_ids = [x.strip() for x in f.readlines()]

        
    #test_book_ids = test_book_ids[:10]
    print(len(test_book_ids), 'books')
    
    
    func = partial(process_book, sent_dir, para_to_sent_dir)
    
    pool = multiprocessing.Pool(processes=32)
    data = pool.map(func, test_book_ids)
    pool.close()
    pool.join()
    
    df = pd.DataFrame(data)
    df.rename(columns={0:'book_id', 1:'status'}, inplace=True)
    df.to_csv('log_test_tok_para_to_sent.csv', index=False)
    print('Done!')
    
    
    
