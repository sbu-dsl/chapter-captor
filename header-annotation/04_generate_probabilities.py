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

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def get_scores(tokens, model, device, tokenizer, sequence_length, slide_by):
    
    # Convert tokens to IDs
    chunk_list = list(chunks(tokens, 512))
    toks = list()
    for c in chunk_list:
        toks += tokenizer.convert_tokens_to_ids(c)
    
    #toks = tokenizer.convert_tokens_to_ids(tokens)
    
    # Generate test sequences using sliding window
    test_sequences = list()
    test_labels_dummy = list()
    test_token_indices = list()
    
    idx = 0
    end_flag = False
    
    while idx < len(toks):
        if not end_flag and idx + sequence_length >= len(toks):
            idx = len(toks) - sequence_length
            end_flag = True
        # Get window
        s = toks[idx:idx + sequence_length]
        test_sequences.append(s)
        test_labels_dummy.append([0 for _ in s])
        test_token_indices.append([elem for elem in range(idx, idx + sequence_length)])
        idx += slide_by
        if end_flag:
            break
    
    # Get predictions for test sequences
    batch_size = 32
    prediction_inputs = torch.tensor(test_sequences)
    prediction_labels_dummy = torch.tensor(test_labels_dummy)
    
    prediction_data = TensorDataset(prediction_inputs, prediction_labels_dummy)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size, num_workers=1)
    
    model.eval()
    
    predictions = list()
    
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_labels_dummy = batch
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None)
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    
    # Flatten output
    flat_preds = list()
    for batch in predictions:
        # batch is 8 x 100 x 2
        for sequence in batch:
            # sequence is 100 x 2
            for probs in sequence:
                # probs is 1 x 2
                flat_preds.append(probs)
    
    flat_probabilities = list()
    for x in flat_preds:
        tmp0 = np.exp(x[0])
        tmp1 = np.exp(x[1])
        summ = tmp0 + tmp1
        flat_probabilities.append(tmp1 / summ)
    
    flat_token_indices = [item for sublist in test_token_indices for item in sublist]
    
    d_probs = dict()
    
    for iterator in range(len(flat_token_indices)):
        index = flat_token_indices[iterator]

        if index in d_probs:
            d_probs[index].append(flat_probabilities[iterator])
        else:
            d_probs[index] = [flat_probabilities[iterator]]
    
    new_probs = [0] * (max(d_probs.keys()) + 1)
    for idx in d_probs.keys():
        new_probs[idx] = max(d_probs[idx])
    
    return new_probs
    

def generate_header_probabilities_from_model(input_dir, output_dir, seq_len, model, device, tokenizer, book_id):
    try:
        base_path = os.path.join(input_dir, book_id + '.xml')
        header_csv_path = os.path.join(output_dir, book_id + '.csv')
        output_xml_path = os.path.join(output_dir, book_id + '_lines.xml')

        # Read from input XML
        parser = ET.XMLParser(huge_tree=True)
        tree = ET.parse(str(base_path), parser=parser)
        book = tree.getroot()

        # Get content from front matter and body (if present)
        front_matter = book.find('front')
        body = book.find('body')

        assert body is not None

        # Convert to lines format XML
        line_number = 0
        if front_matter is not None:
            if front_matter.text is not None:
                front_matter_lines = front_matter.text.splitlines(keepends=True)
                front_matter.text = None
                previous = None
                for elem in front_matter_lines:
                    e = ET.Element("line")
                    e.text = elem
                    e.set("num", str(line_number))
                    if previous is None:
                        front_matter.insert(0, e)
                    else:
                        previous.addnext(e)
                    previous = e
                    line_number += 1
            for child in front_matter.getchildren():
                if child.tail is not None:
                    child_lines = child.tail.splitlines(keepends=True)
                    child.tail = None
                    previous = child
                    for elem in child_lines:
                        e = ET.Element("line")
                        e.text = elem
                        e.set("num", str(line_number))
                        previous.addnext(e)
                        previous = e
                        line_number += 1

        if body.text is not None:
            body_lines = body.text.splitlines(keepends=True)
            body.text = None
            previous = None
            for elem in body_lines:
                e = ET.Element("line")
                e.text = elem
                e.set("num", str(line_number))
                if previous is None:
                    body.insert(0, e)
                else:
                    previous.addnext(e)
                previous = e
                line_number += 1
        for child in body.getchildren():
            if child.tail is not None:
                child_lines = child.tail.splitlines(keepends=True)
                child.tail = None
                previous = child
                for elem in child_lines:
                    e = ET.Element("line")
                    e.text = elem
                    e.set("num", str(line_number))
                    previous.addnext(e)
                    previous = e
                    line_number += 1


        content = [x.text for x in book.findall(".//line")]


        # Generate probabilities per line

        # Convert to tokens
        matrix = list()
        for line_number, line in enumerate(content):
            text = line.replace('\n', ' [unused1] ')
            doc_tokens = list()
            char_to_word_offset = list()
            prev_is_whitespace = True
            for c in text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
            tok_to_orig_index = list()
            all_doc_tokens = list()
            for i, token in enumerate(doc_tokens):
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
            doc_tokens_to_line_index = [char_to_word_offset.index(x) for x in range(len(doc_tokens))]

            for final_token_idx, final_token in enumerate(all_doc_tokens):
                matrix.append([final_token, line_number, doc_tokens_to_line_index[tok_to_orig_index[final_token_idx]]])

        df = pd.DataFrame(matrix)
        df.rename(columns={0:'token', 1:'line_number', 2:'token_word_pos'}, inplace=True)

        token_list = list(df['token'].apply(lambda x: str(x)))

        probs = get_scores(token_list, model, device, tokenizer, seq_len, seq_len // 2)

        df['prob'] = probs

        df = df.groupby(['line_number', 'token_word_pos'], as_index=False).agg({'token': (lambda x: ''.join([y[2:] if y.startswith('##') else y for y in x])), 'prob': 'mean'})

        df = df[['token', 'line_number', 'token_word_pos', 'prob']]

        df.to_csv(header_csv_path, index=False)

        with open(output_xml_path, 'wb') as f:
            f.write(ET.tostring(book, pretty_print=True))

        return book_id, 0
    
    except:
        return book_id, -1


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
    
    # Read location where model checkpoints are stored
    checkpoint_dir = config.get('03_Train_model', 'checkpoint_dir')
    if not os.path.isdir(checkpoint_dir):
        print('Please run 03_train_model.py first.')
        exit()
    
    # Read number of epochs for which model was trained
    num_epochs = int(config.get('03_Train_model', 'num_epochs'))
    
    # Read test books location
    base_xml_files_dir = config.get('04_Generate_test_probs', 'base_xml_files_dir')
    if not os.path.isdir(base_xml_files_dir):
        print('Please provide a valid directory name where the xml files for the training set books are stored, in the "base_xml_files_dir" field.')
        exit()
    
    # Read location to store probability outputs
    prob_dir = config.get('04_Generate_test_probs', 'prob_dir')
    if not os.path.isdir(prob_dir):
        os.makedirs(prob_dir)
    
    # Read sequence length
    seq_len = int(config.get('02_Generate_training_sequences', 'seq_len'))
    
    # Read location to store status of header extraction
    log_file = config.get('04_Generate_test_probs', 'log_file')
    
    
    # BERT model
    model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=2)
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'epoch_' + str(num_epochs) + '.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, never_split=['[unused1]'])
    
    
    l = list()
    print(len(test_book_ids), 'books')
    for idx, book_id in enumerate(test_book_ids):
        print(idx, book_id)
        book, status = generate_header_probabilities_from_model(base_xml_files_dir, prob_dir, seq_len, model, device, tokenizer, book_id)
        l.append((book, status))
    
    print('Done! Saving status results to log file...')
        
    df = pd.DataFrame(l, columns=['bookID', 'status'])
    
    df.to_csv(log_file, index=False)
    
    print('Saved results to log file!')
    
    
    
    