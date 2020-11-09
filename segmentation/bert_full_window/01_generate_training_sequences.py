import pandas as pd
import lxml.etree as ET
import random

from transformers import BertTokenizer

def get_examples_book(pred_dir, tokenizer, book_id):
    try:
        filename = pred_dir + str(book_id) + '.xml'
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
        
        start_para_nums.append(int(b.findall('.//p')[-1].attrib['num']) + 1)
        examples = list()
        prev_section_start = -1
        for idx, p_num in enumerate(start_para_nums[:-1]):
            
            # Positive example
            prev_tokens = list()
            prev_idx = p_num - 1
            while prev_idx >= prev_section_start and prev_idx >= 0 and len(prev_tokens) < 254:
                prev_elem = b.find('.//p[@num=\'' + str(prev_idx) + '\']')
                prev_text = ' '.join([x.text for x in prev_elem.findall('.//s')])
                tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(prev_text)))
                prev_tokens = tokens + prev_tokens
                prev_tokens = prev_tokens[-254:]
                prev_idx -= 1
            
            next_section_start = start_para_nums[idx + 1]
            next_tokens = list()
            next_idx = p_num
            while next_idx < next_section_start and len(next_tokens) < 254:
                next_elem = b.find('.//p[@num=\'' + str(next_idx) + '\']')
                next_text = ' '.join([x.text for x in next_elem.findall('.//s')])
                tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(next_text)))
                next_tokens = next_tokens + tokens
                next_tokens = next_tokens[:254]
                next_idx += 1
            
            examples.append([prev_tokens, next_tokens, 1])
            
            
            # Previous chapter
            if prev_section_start != -1:
                idx_use = random.randint(prev_section_start, p_num - 2)
                
                prev_tokens = list()
                prev_idx = idx_use - 1
                while prev_idx >= prev_section_start and prev_idx >= 0 and len(prev_tokens) < 254:
                    prev_elem = b.find('.//p[@num=\'' + str(prev_idx) + '\']')
                    prev_text = ' '.join([x.text for x in prev_elem.findall('.//s')])
                    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(prev_text)))
                    prev_tokens = tokens + prev_tokens
                    prev_tokens = prev_tokens[-254:]
                    prev_idx -= 1

                next_section_start = start_para_nums[idx + 1]
                next_tokens = list()
                next_idx = idx_use
                while next_idx < p_num and len(next_tokens) < 254:
                    next_elem = b.find('.//p[@num=\'' + str(next_idx) + '\']')
                    next_text = ' '.join([x.text for x in next_elem.findall('.//s')])
                    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(next_text)))
                    next_tokens = next_tokens + tokens
                    next_tokens = next_tokens[:254]
                    next_idx += 1
                
                examples.append([prev_tokens, next_tokens, 0])
            prev_section_start = p_num
            
            # Next chapter
            idx_use = random.randint(p_num + 1, start_para_nums[idx + 1] - 2)
                
            prev_tokens = list()
            prev_idx = idx_use - 1
            while prev_idx >= p_num and prev_idx >= 0 and len(prev_tokens) < 254:
                prev_elem = b.find('.//p[@num=\'' + str(prev_idx) + '\']')
                prev_text = ' '.join([x.text for x in prev_elem.findall('.//s')])
                tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(prev_text)))
                prev_tokens = tokens + prev_tokens
                prev_tokens = prev_tokens[-254:]
                prev_idx -= 1

            next_section_start = start_para_nums[idx + 1]
            next_tokens = list()
            next_idx = idx_use
            while next_idx < start_para_nums[idx + 1] and len(next_tokens) < 254:
                next_elem = b.find('.//p[@num=\'' + str(next_idx) + '\']')
                next_text = ' '.join([x.text for x in next_elem.findall('.//s')])
                tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(next_text)))
                next_tokens = next_tokens + tokens
                next_tokens = next_tokens[:254]
                next_idx += 1

            examples.append([prev_tokens, next_tokens, 0])
        return examples
    except:
        return []



if __name__ == '__main__':
    # Use appropriate locations
    train_books_list_file = 'train_books.txt'
    pred_dir = 'use_books_sentencized/'
    save_loc = 'train_sequences_tokenized.csv'
    
    with open(train_books_list_file, 'r') as f:
        train_book_ids = [x.strip() for x in f.readlines()]

    
    print(len(train_book_ids), 'books')
    
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    
    training_data = list()
    
    for idx, book_id in enumerate(train_book_ids):
        training_data += get_examples_book(pred_dir, tokenizer, book_id)
        if idx % 100 == 0:
            print(idx, len(training_data))
        
    df = pd.DataFrame(training_data)
    
    df.rename(columns={0:'para1_tokens', 1:'para2_tokens', 2:'label'}, inplace=True)
    
    df['para1_len'] = df['para1_tokens'].apply(lambda x: len(x))
    df['para2_len'] = df['para2_tokens'].apply(lambda x: len(x))
    
    df.to_csv(save_loc, index=False)
        
        
