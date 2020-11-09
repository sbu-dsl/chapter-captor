import argparse
import configparser
import os
import lxml.etree as ET
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertForTokenClassification
import numpy as np
import tensorflow as tf
import pickle

from functools import partial
import multiprocessing

from header_util import get_header_lines
from regex_util import get_rules


import time
import timeout_decorator

@timeout_decorator.timeout(60)
def annotate_headers(input_dir, output_xml_dir, output_pickle_dir, regex_rules_cache, book_id):
    input_xml_path = os.path.join(input_dir, book_id + '_lines.xml')
    header_csv_path = os.path.join(input_dir, book_id + '.csv')
    output_xml_path = os.path.join(output_xml_dir, book_id + '_headers.xml')

    if os.path.exists(output_xml_path):
        return book_id, 1

    print('started', book_id)

    # Read from input XML
    parser = ET.XMLParser(huge_tree=True)
    tree = ET.parse(str(input_xml_path), parser=parser)
    book = tree.getroot()

    body = book.find('body')
    body_start = int(body.find('.//line').attrib['num'])

    # Read token-wise predictions
    toks_df = pd.read_csv(header_csv_path)

    # Read regex rules from cache
    with open(regex_rules_cache, 'rb') as f:
        all_seqs, all_seqs_orig, rules, priority = pickle.load(f)

    output_pickle_prefix = os.path.join(output_pickle_dir, book_id)
    # Get list of header line numbers
    header_lines = get_header_lines(toks_df, all_seqs, all_seqs_orig, rules, priority, body_start, output_pickle_prefix, book)


    # First line: last line
    d = {x[0]:x[-1] for x, y in header_lines}
    # First line: attributes
    attrs = {x[0]:y for x, y in header_lines}

    # Enclose line numbers in header tags contained in attrs

    # Delete section tags
    ET.strip_tags(book, "section")

    for from_line, to_line in d.items():
        desc, number, number_text, number_type, title, rule_text = attrs[from_line]
        f = book.find('.//line[@num="' + str(from_line) + '"]')

        new_element = ET.Element('header')
        new_element.set('desc', str(desc))
        new_element.set('number', str(number))
        new_element.set('number_text', str(number_text))
        new_element.set('number_type', str(number_type))
        new_element.set('title', str(title))
        new_element.set('rule_text', str(rule_text).strip(','))

        prev = f.getprevious()
        if prev is not None:
            for line_num in range(from_line, to_line + 1):
                e = book.find('.//line[@num="' + str(line_num) + '"]')
                new_element.append(e)
            prev.addnext(new_element)
        else:
            parent = f.getparent()
            for line_num in range(from_line, to_line + 1):
                e = book.find('.//line[@num="' + str(line_num) + '"]')
                new_element.append(e)
            parent.insert(0, new_element)


    ET.strip_tags(book, "line")

    # Write to file
    with open(output_xml_path, 'wb') as f:
        f.write(ET.tostring(book, pretty_print=True))

    return book_id, 0
    
    
def process_book(input_dir, output_xml_dir, output_pickle_dir, regex_rules_cache, book_id):
    return annotate_headers(input_dir, output_xml_dir, output_pickle_dir, regex_rules_cache, book_id)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))
    
    # Read set of test books
    # Read the list of book IDs in the test set
    test_set_books_file = config.get('04_Generate_test_probs', 'test_books_list')
    if not os.path.isfile(test_set_books_file):
        print('Please provide a valid file name for the list of test set book IDs in the "test_books_list" field.')
        exit()
    with open(test_set_books_file) as f:
        test_book_ids = [x.strip() for x in f.readlines()]
    
    # Read location to store probability outputs
    input_dir = config.get('04_Generate_test_probs', 'prob_dir')
    if not os.path.isdir(input_dir):
        print('Please run 04_generate_test_probs.py first.')
        exit()
        
    # Read location to store annotated XML output
    output_xml_dir = config.get('05_Annotate_headers', 'output_xml_dir')
    if not os.path.isdir(output_xml_dir):
        os.makedirs(output_xml_dir)
    
    # Read location to store staged header output
    output_pickle_dir = config.get('05_Annotate_headers', 'output_pickle_dir')
    if not os.path.isdir(output_pickle_dir):
        os.makedirs(output_pickle_dir)
    
    
    # Read number of processes to use
    num_procs = int(config.get('05_Annotate_headers', 'num_procs'))
    
    # Read location to store status of header extraction
    log_file = config.get('05_Annotate_headers', 'log_file')
    
    
    
    regex_rules_cache = './regex_rules_cache.pkl'
    # Generating regex rules
    if not os.path.exists(regex_rules_cache):
        print("Generating regex rules...")
        all_seqs, all_seqs_orig, rules, priority = get_rules()
        with open(regex_rules_cache, 'wb') as f:
            pickle.dump((all_seqs, all_seqs_orig, rules, priority), f)
    
    
    func = partial(process_book, input_dir, output_xml_dir, output_pickle_dir, regex_rules_cache)
    
    pool = multiprocessing.Pool(processes=num_procs)
    data = pool.map(func, test_book_ids)
    pool.close()
    pool.join()
    
    print('Done! Saving status results to log file...')

    df = pd.DataFrame(data, columns=['bookID', 'status'])
    df.to_csv(log_file, index=False)
    
    print('Saved results to log file!')
