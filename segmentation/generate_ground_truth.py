import pandas as pd
import lxml.etree as ET
import random
import os

import pickle
import multiprocessing
from functools import partial

def get_examples_book(input_dir, output_dir, book_id):
        filename = input_dir + str(book_id) + '.xml'
        parser = ET.XMLParser(huge_tree=True)
        tree = ET.parse(filename, parser=parser)
        book = tree.getroot()
        b = book.find('.//body')
        
        headers = b.findall('.//header')
        
        start_para_nums = list()
        for h in headers:
            t = h.getnext()
            if t.tag == 'p':
                start_para_nums.append(int(t.attrib['num']))
        
        gt = list()
        for para_num in start_para_nums:
            last_p = b.find('.//p[@num=\'' + str(para_num - 1) + '\']')
            if last_p is None:
                continue
            sents = last_p.findall('.//s')
            num = int(sents[-1].attrib['num'])
            gt.append(num)
        
        max_sent_num = b.findall('.//s')[-1].attrib['num']
        
        with open(os.path.join(output_dir, book_id + '_gt_sents.pkl'), 'wb') as f:
            pickle.dump(gt, f)
        with open(os.path.join(output_dir, book_id + '_max_sent_num.pkl'), 'wb') as f:
            pickle.dump(max_sent_num, f)

        return book_id, 0

if __name__ == '__main__':
    # Use appropriate locations
    test_books_list_file = 'test_books.txt'
    
    sent_dir = 'use_books_sentencized/'
    output_dir = 'test_gt_sentences/'
    
    with open(test_books_list_file, 'r') as f:
        test_book_ids = [x.strip() for x in f.readlines()]

        
    print(len(test_book_ids), 'books')
    
    func = partial(get_examples_book, sent_dir, output_dir)
    
    pool = multiprocessing.Pool(processes=32)
    data = pool.map(func, test_book_ids)
    pool.close()
    pool.join()
    
    df = pd.DataFrame(data)
    df.rename(columns={0:'book_id', 1:'status'}, inplace=True)
    df.to_csv('log.csv', index=False)
    print('Done!')
