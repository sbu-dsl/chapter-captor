import argparse
import configparser
import os
import nltk
import gzip
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
import random
import multiprocessing
import nltk
import pickle

import multiprocessing
from functools import partial


# Function to remove Gutenberg-specific header and footer
def remove_gutenberg_header_footer(lines):
    start_arr = [idx for idx in range(len(lines)) if '***' in lines[idx] and 'START' in lines[idx].upper() and 'GUTENBERG' in lines[idx].upper()]
    end_arr = [idx for idx in range(len(lines)) if '***' in lines[idx] and 'END' in lines[idx].upper() and 'GUTENBERG' in lines[idx].upper()]
    
    if len(start_arr) > 0 and len(end_arr) > 0:
        return lines[start_arr[0] + 1 : end_arr[0]]
    elif len(start_arr) > 0:
        return lines[start_arr[0] + 1:]
    elif len(end_arr) > 0:
        return lines[:end_arr[0]]
    return lines

# Function to obtain line numbers for annotated headers from text file
def get_annotated_headers(book_index, text_file_dir, header_content_dir):
    def find_index(heading, book_lines):
        ans = list()
        idx = 0
        partial_match = 0
        curr = ''
        while idx < len(book_lines):
            if len(book_lines[idx]) == 0:
                if partial_match > 0:
                    partial_match += 1
                idx += 1
                continue

            if partial_match > 0:
                curr += ' ' + book_lines[idx]
                curr = ' '.join(nltk.word_tokenize(curr))
                if curr == heading:
                    ans.append((start_idx, partial_match + 1))
                    partial_match = 0
                    curr = ''
                elif heading.startswith(curr):
                    partial_match += 1
                else:
                    partial_match = 0
                    curr = ''

            elif book_lines[idx] == heading:
                ans.append((idx, 1))

            elif heading.startswith(book_lines[idx]):
                curr += book_lines[idx]
                partial_match = 1
                start_idx = idx

            idx += 1
        return ans

    lines = list()
    with gzip.open(text_file_dir + book_index + '.txt.gz', 'rt') as f:
        for line in f:
            lines.append(line)

    lines = remove_gutenberg_header_footer(lines)
    mod = [' '.join(nltk.word_tokenize((str(x)).strip())) for x in lines]

    df = pd.read_csv(header_content_dir + book_index + '.csv')
    if 'token' in df.columns:
        df['text'] = df['token']
    hc = list()
    for idx, row in df.iterrows():
        if row['label'] == 'H':
            header = row['text']
            if idx + 1 < len(df):
                content = df['text'][idx + 1]
            else:
                content = ' '
            hc.append((' '.join(nltk.word_tokenize(str(header).strip())), nltk.word_tokenize(str(content).strip())[0]))
    header_list = list()
    for header, content in hc:
        l = find_index(header, mod)
        for index, length in l:
            content_index = index + length
            while len(mod[content_index]) == 0:
                content_index += 1
            if mod[content_index].startswith(content):
                header_list.append((index, length))
            else:
                pass
    return lines, header_list

# Function to generate training sequences from a book
def get_sequences_whitespace(lines, headers, seq_len, tokenizer):
    token_sequences = list()
    label_sequences = list()

    for index, length in headers:
        text = ' '.join(lines[index:index + length])
        tokens = tokenizer.tokenize(text.replace('\n', ' [unused1] '))

        need_tokens = seq_len - len(tokens)
        
        prev_tokens_needed = random.randint(0, need_tokens)
        next_tokens_needed = need_tokens - prev_tokens_needed

        # Generate previous tokens
        prev_tokens = list()
        idx = index - 1
        max_count = seq_len * 2
        while True:
            if idx < 0 or len(prev_tokens) >= prev_tokens_needed or max_count == 0:
                prev_tokens = prev_tokens[-prev_tokens_needed:]
                break
            try:
                prev_tokens = tokenizer.tokenize(lines[idx].replace('\n', ' [unused1] ')) + prev_tokens
            except:
                pass
            idx -= 1
            max_count -= 1

        # Generate next tokens
        next_tokens = list()
        idx = index + length
        max_count = seq_len * 2
        while True:
            if idx >= len(lines) or len(next_tokens) >= next_tokens_needed or max_count == 0:
                next_tokens = next_tokens[:next_tokens_needed]
                break
            try:
                next_tokens = next_tokens + tokenizer.tokenize(lines[idx].replace('\n', ' [unused1] '))
            except:
                pass
            idx += 1
            max_count -= 1

        ts = prev_tokens + tokens + next_tokens
        ls = [0] * len(prev_tokens) + [1] * len(tokens) + [0] * len(next_tokens)

        if len(ts) == seq_len:
            token_sequences.append(tokenizer.convert_tokens_to_ids(ts))
            label_sequences.append(ls)

    return token_sequences, label_sequences


def process_book(text_files_dir, header_content_dir, seq_gen_dir, seq_len, tokenizer, book_index):
    try:
        lines, headers = get_annotated_headers(book_index, text_files_dir, header_content_dir)
        token_sequences, label_sequences = get_sequences_whitespace(lines, headers, seq_len, tokenizer)

        with open(os.path.join(seq_gen_dir, book_index + '_tokens.pkl'), 'wb') as f:
            pickle.dump(token_sequences, f)
        with open(os.path.join(seq_gen_dir, book_index + '_labels.pkl'), 'wb') as f:
            pickle.dump(label_sequences, f)
        return book_index, 0
    
    except Exception as e:
        print(e)
        return book_index, -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))
    
    # Read the list of book IDs in the training set
    train_set_books_file = config.get('02_Generate_training_sequences', 'train_books_list')
    if not os.path.isfile(train_set_books_file):
        print('Please provide a valid file name for the list of training set book IDs in the "train_books_list" field.')
        exit()
    with open(train_set_books_file) as f:
        train_book_ids = [x.strip() for x in f.readlines()]
        
    
    # Read the directory which contains txt.gz files
    text_files_dir = config.get('02_Generate_training_sequences', 'text_files_dir')
    if not os.path.isdir(text_files_dir):
        print('Please provide a valid directory name where the txt.gz files for the training set books are stored, in the "text_files_dir" field.')
        exit()
    
    # Read the directory where the extracted headers are stored
    header_content_dir = config.get('01_Extract_headers_from_HTML', 'extracted_header_dir')
    if not os.path.isdir(extracted_header_dir):
        print('Please run 01_extract_headers_from_html.py first.')
        exit()
    
    # Read the directory where extracted training sequences are to be stored
    seq_gen_dir = config.get('02_Generate_training_sequences', 'generated_sequence_dir')
    if not os.path.isdir(seq_gen_dir):
        os.makedirs(seq_gen_dir)
    
    # Read the sequence length to be generated
    seq_len = int(config.get('02_Generate_training_sequences', 'seq_len'))
    
    # Read number of processes to use
    num_procs = int(config.get('02_Generate_training_sequences', 'num_procs'))
    
    # Read location to store status of header extraction
    log_file = config.get('02_Generate_training_sequences', 'log_file')
    
    # Define tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, never_split=['[unused1]'])


    func = partial(process_book, text_files_dir, header_content_dir, seq_gen_dir, seq_len, tokenizer)

    pool = multiprocessing.Pool(processes=num_procs)
    data = pool.map(func, train_book_ids)
    pool.close()
    pool.join()
    
    print('Done! Saving status results to log file...')

    df = pd.DataFrame(data, columns=['bookID', 'status'])
    df.to_csv(log_file, index=False)
    
    print('Saved results to log file!')