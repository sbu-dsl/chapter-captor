import pandas as pd
import lxml.etree as ET
import random

def get_examples_book(pred_dir, book_id):
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
        prev_start = None
        for idx, p_num in enumerate(start_para_nums[:-1]):
            # This and previous
            prev_elem = b.find('.//p[@num=\'' + str(p_num - 1) + '\']')
            curr_elem = b.find('.//p[@num=\'' + str(p_num) + '\']')
            if prev_elem is None or curr_elem is None:
                continue
            prev_text = ' '.join([x.text for x in prev_elem.findall('.//s')])
            curr_text = ' '.join([x.text for x in curr_elem.findall('.//s')])
            examples.append([prev_text, curr_text, 1])
            
            if prev_start is not None:
                # Find a random number >= prev_start and < p_num - 1
                idx_use = random.randint(prev_start, p_num - 2)
                prev_elem = b.find('.//p[@num=\'' + str(idx_use) + '\']')
                curr_elem = b.find('.//p[@num=\'' + str(idx_use + 1) + '\']')
                prev_text = ' '.join([x.text for x in prev_elem.findall('.//s')])
                curr_text = ' '.join([x.text for x in curr_elem.findall('.//s')])
                examples.append([prev_text, curr_text, 0])
            prev_start = p_num
            
            next_start = start_para_nums[idx + 1]
            # Find a random number >= p_num and < start_para_nums[idx + 1] - 1
            idx_use = random.randint(p_num, start_para_nums[idx + 1] - 2)
            prev_elem = b.find('.//p[@num=\'' + str(idx_use) + '\']')
            curr_elem = b.find('.//p[@num=\'' + str(idx_use + 1) + '\']')
            prev_text = ' '.join([x.text for x in prev_elem.findall('.//s')])
            curr_text = ' '.join([x.text for x in curr_elem.findall('.//s')])
            examples.append([prev_text, curr_text, 0])
        return examples
    except:
        return []



if __name__ == '__main__':
    # Use appropriate locations
    train_books_list_file = 'train_books.txt'
    pred_dir = 'use_books_sentencized/'
    save_loc = 'train_sequences.csv'
    
    with open(train_books_list_file, 'r') as f:
        train_book_ids = [x.strip() for x in f.readlines()]

    
    print(len(train_book_ids), 'books')
    
    training_data = list()
    
    for idx, book_id in enumerate(train_book_ids):
        training_data += get_examples_book(pred_dir, book_id)
        if idx % 100 == 0:
            print(idx, len(training_data))
        
    df = pd.DataFrame(training_data)
    
    df.rename(columns={0:'para1', 1:'para2', 2:'label'}, inplace=True)
    
    df.to_csv(save_loc, index=False)
        
        
