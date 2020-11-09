import lxml.etree as ET
import os
import glob

from stanfordnlp.server import CoreNLPClient

import pickle

import pandas as pd

from functools import partial
import multiprocessing


def sentencize(header_annot_dir, client, book_id):
    
    filename = os.path.join(header_annot_dir, str(book_id) + '.xml')
    
    parser = ET.XMLParser(huge_tree=True)
    tree = ET.parse(filename, parser=parser)
    book = tree.getroot()
    
    para_end_sentences = list()
    lemma_dict = dict()
    
    start_sentence_number = 0
    
    b = book.find('.//body')
    
    header_elems = [x for idx, x in enumerate(b)]
    
    for idx, element in enumerate(header_elems):
        content = element.tail
        element.tail = ""
        if content is None:
            continue
        
        ann = client.annotate(content)
        
        init_offset = content.index(ann.sentence[0].token[0].originalText)
        
        prev = element
        
        for idx_2, sent in enumerate(ann.sentence):
            
            sentence_tag = ET.Element("s")
            sentence_tag.text = content[init_offset + sent.characterOffsetBegin:init_offset + sent.characterOffsetEnd]
            num = start_sentence_number + idx_2
            sentence_tag.set('num', str(num))
            
            if sent.token[-1].after.startswith('\n\n') or idx_2 == len(ann.sentence) - 1:
                para_end_sentences.append(num)
            
            lemma_dict[num] = [tok.lemma for tok in sent.token]
            
            prev.addnext(sentence_tag)
            prev = sentence_tag
        
        start_sentence_number += len(ann.sentence)
    
    if len(para_end_sentences) == 0:
        para_end_sentences = [num]
    if para_end_sentences[-1] != num:
        para_end_sentences.append(num)
    
    tree = ET.ElementTree(book)
    
    return tree, para_end_sentences, lemma_dict


def paragraphize(tree, para_end_sentences):
    book = tree.getroot()
    
    body = book.find('.//body')
    
    elems = [x for x in body]
    
    new_body = ET.Element('body')
    
    para_num = 0
    start = 0
    end = para_end_sentences[0]
    
    for elem in elems:
        if elem.tag == 'header':
            new_body.append(elem)
            continue
        num = int(elem.get('num'))
        if num == start:
            current_para = ET.Element('p')
            current_para.set('num', str(para_num))
        if num >= start and num <= end:
            current_para.append(elem)
        if num == end:
            new_body.append(current_para)
            para_num += 1
            start = end + 1
            if para_num < len(para_end_sentences):
                end = para_end_sentences[para_num]
            else:
                end = None
    
    idx = [idx for idx, elem in enumerate(book) if elem.tag == 'body'][0]
    book[idx] = new_body
    
    tree = ET.ElementTree(book)
    
    return tree


def process_book(header_annot_dir, lemma_dir, tree_dir, book_id):
    
    if os.path.exists(os.path.join(tree_dir, book_id + '.xml')) and os.path.exists(os.path.join(lemma_dir, book_id + '.pkl')):
        return book_id, 'Exists'
    
    os.environ["CORENLP_HOME"] = "~/stanford_corenlp/stanford-corenlp-full-2018-10-05"
    
    try:
        with CoreNLPClient(annotators=['tokenize','lemma'], timeout=30000, max_char_length=100000000, be_quiet=True, start_server=False) as client:
            tree, para_end_sentences, lemma_dict = sentencize(header_annot_dir, client, book_id)

        tree2 = paragraphize(tree, para_end_sentences)

        filename = os.path.join(tree_dir, book_id + '.xml')
        tree2.write(filename, pretty_print=True)

        with open(os.path.join(lemma_dir, book_id + '.pkl'), 'wb') as f:
            pickle.dump(lemma_dict, f)
    except Exception as e:
        print(book_id, e)
        return book_id, e
    
    print(book_id, 'Success!')
    return book_id, 'Success'


if __name__ == "__main__":
    
    # Use appropriate locations
    header_annot_dir = 'annot_header_dir/'
    
    lemma_dir = 'lemmas/'
    tree_dir = 'sentencized/'
    
    if not os.path.exists(lemma_dir):
        os.makedirs(lemma_dir)
    if not os.path.exists(tree_dir):
        os.makedirs(tree_dir)
    
    with open('train_book_ids.txt', 'r') as f:
        books = f.read().splitlines()

    with open('test_book_ids_seg.txt', 'r') as f:
        books += f.read().splitlines()

    func = partial(process_book, header_annot_dir, lemma_dir, tree_dir)
    
    pool = multiprocessing.Pool(processes=50)
    data = pool.map(func, books)
    pool.close()
    pool.join()
    
    df = pd.DataFrame(data)
    df.rename(columns={0:'book_id', 1:'status'}, inplace=True)
    df.to_csv('./log_file.csv', index=False)
    
    print('Done!')



