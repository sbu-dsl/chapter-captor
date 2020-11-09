import argparse
import configparser
import os
import requests
from bs4 import BeautifulSoup, NavigableString, Comment
import re
import glob
import gzip
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import os.path
from functools import partial
import multiprocessing

# Function to strip unwanted tags (such as italics) from text
def strip_tags(html, invalid_tags):
    soup = BeautifulSoup(html, features='lxml')
    for tag in soup.findAll(True):
        if tag.name in invalid_tags:
            s = ""
            for c in tag.contents:
                if not isinstance(c, NavigableString):
                    c = strip_tags(str(c), invalid_tags)
                s += str(c)
            tag.replaceWith(s)
    return soup

# Class to store a header-content pair
class HeaderContent(object):
    def __init__(self, header, content):
        self.header = header
        self.content = content

    def add_header(self, header):
        self.header.append(header)

    def add_paragraph(self, paragraph):
        self.content.append(paragraph)

    def get_num_headers(self):
        return len(self.header)

    def get_num_paras(self):
        return len(self.content)

    def get_num_words(self):
        return len(' '.join([x.strip() for x in self.content]).split())

    def print_headers(self):
        for elem in self.header:
            print(elem.strip())

    def print_joined_headers(self):
        print(' '.join([elem.strip().replace('\n', ' ') for elem in self.header]))

    def print_short_content(self):
        for elem in self.content:
            print(elem.strip()[:20])

# Function to obtain list of HeaderContent objects
def segment_book(html_location):
    url = html_location
    with open(url, 'r') as f:
        html = f.read()
    
    soup = strip_tags(html, ['b', 'i', 'u'])
    
    book = list()
    prev_header = False
    curr_obj = HeaderContent(header=list(), content=list())

    for x in soup.find_all('span', {'class': 'pagenum'}):
        x.decompose()
    for x in soup.find_all('span', {'class': 'returnTOC'}):
        x.decompose()
    for x in soup.find_all(attrs={'class': 'figcenter'}):
        x.decompose()
    for x in soup.find_all(attrs={'class': 'caption'}):
        x.decompose()
    for x in soup.find_all(attrs={'class': 'totoc'}):
        x.decompose()
    for x in soup.find_all(['pre', 'img', 'style']):
        x.decompose()
    for x in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        for elem in x.find_all():
            if elem.name == 'i':
                elem.decompose()

    skip_count = 0
    for x in soup.find_all():
        if x.name == 'html':
            continue

        if skip_count > 0:
            skip_count -= 1
            continue

        if x.name == 'hr' and x.has_attr('class') and 'pb' in x['class']:
            book.append(curr_obj)
            curr_obj = HeaderContent(header=list(), content=list())

        if x.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if prev_header:
                curr_obj.add_header(x.text)
            else:
                book.append(curr_obj)
                curr_obj = HeaderContent(header=list(), content=list())
                curr_obj.add_header(x.text)
                prev_header = True
            skip_count = len(x.find_all())
        else:
            t = ''.join([elem for elem in x.find_all(text=True, recursive=False) if not isinstance(elem, Comment)]).strip()
            
            if t:
                if 'start of the project gutenberg' in ' '.join(t.lower().split()):
                    book = list()
                    curr_obj = HeaderContent(header=list(), content=list())
                    continue
                if 'end of the project gutenberg' in ' '.join(t.lower().split()):
                    break
                if 'xml' in t and 'version' in t and 'encoding' in t:
                    continue
                curr_obj.add_paragraph(t)
                prev_header = False
    book.append(curr_obj)
    return book


def process_chapter(header_contents):
    retval = list()
    
    l = list()
    for elem in header_contents:
        headers = elem.header
        if len(headers) > 0 and headers[0].strip().lower().startswith('chapter'):
            l.append(len(headers))
    max_count = max(set(l), key=l.count)

    for elem in header_contents:
        headers = elem.header
        found = False
        for idx in range(len(headers)):
            h = headers[idx]
            if idx < len(headers) - max_count:
                # Append as content 'C'
                retval.append(('C', h))
            elif found:
                # Append words with label 'H'
                retval.append(('H', h))
            elif h.strip().lower().startswith('chapter'):
                # Append this and all subsequent headers with label 'H'
                found = True
                retval.append(('H', h))
            else:
                retval.append(('C', h))
        contents = elem.content
        for c in contents:
            retval.append(('C', c))

    return retval

def process_part(header_contents):
    retval = list()

    l = list()
    for elem in header_contents:
        headers = elem.header
        if len(headers) > 0 and headers[0].strip().lower().startswith('part'):
            l.append(len(headers))
    max_count = max(set(l), key=l.count)
    
    for elem in header_contents:
        headers = elem.header
        found = False
        for idx in range(len(headers)):
            h = headers[idx]
            if idx < len(headers) - max_count:
                # Append as content 'C'
                retval.append(('C', h))
            elif found:
                # Append words with label 'H'
                retval.append(('H', h))
            elif h.strip().lower().startswith('part'):
                # Append this and all subsequent headers with label 'H'
                found = True
                retval.append(('H', h))
            else:
                retval.append(('C', h))
        contents = elem.content
        for c in contents:
            retval.append(('C', c))

    return retval

def process_roman(header_contents):
    retval = list()
    
    pattern = re.compile("^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$")
    
    l = list()
    for elem in header_contents:
        headers = elem.header
        if len(headers) > 0:
            proc = re.sub(r'([^\s\w]|_)+', ' ', headers[0].strip()).split()
            if len(proc) > 0 and pattern.match(proc[0]):
                l.append(len(headers))
    max_count = max(set(l), key=l.count)
    
    for elem in header_contents:
        headers = elem.header
        found = False
        for idx in range(len(headers)):
            h = headers[idx]
            proc = re.sub(r'([^\s\w]|_)+', ' ', h.strip()).split()
            if idx < len(headers) - max_count:
                # Append as content 'C'
                retval.append(('C', h))
            elif found:
                # Append words with label 'H'
                retval.append(('H', h))
            elif len(proc) > 0 and pattern.match(proc[0]):
                # Append this and all subsequent headers with label 'H'
                found = True
                retval.append(('H', h))
            else:
                retval.append(('C', h))
        contents = elem.content
        for c in contents:
            retval.append(('C', c))

    return retval

def process_number(header_contents):
    retval = list()
    
    l = list()
    for elem in header_contents:
        headers = elem.header
        if len(headers) > 0 and headers[0].strip().lower().replace('.', ' ').split()[0].isnumeric():
            l.append(len(headers))
    max_count = max(set(l), key=l.count)
    
    for elem in header_contents:
        headers = elem.header
        found = False
        for idx in range(len(headers)):
            h = headers[idx]
            if idx < len(headers) - max_count:
                # Append as content 'C'
                retval.append(('C', h))
            elif found:
                # Append words with label 'H'
                retval.append(('H', h))
            elif h.strip().lower().replace('.', ' ').split()[0].isnumeric():
                # Append this and all subsequent headers with label 'H'
                found = True
                retval.append(('H', h))
            else:
                retval.append(('C', h))
        contents = elem.content
        for c in contents:
            retval.append(('C', c))
    return retval

def process_base_case(header_contents):
    retval = list()
    
    if len(header_contents) == 0:
        return retval
    
    if len(header_contents) == 1:
        # return text and labels directly
        headers = header_contents[0].header
        header_text = ''.join([h for h in headers])
        
        contents = header_contents[0].content
        content_text = ''.join([c for c in contents])
        
        return [('H', header_text), ('C', content_text)]

    word_nums = [elem.get_num_words() for elem in header_contents]
    agg = AgglomerativeClustering(n_clusters=2, linkage='average').fit([[x] for x in word_nums])
    cluster_word_arrs = dict()
    cluster_word_arrs[0] = list()
    cluster_word_arrs[1] = list()
    for idx in range(len(word_nums)):
        label = agg.labels_[idx]
        cluster_word_arrs[label].append(word_nums[idx])
    mean_0 = sum(cluster_word_arrs[0]) / len(cluster_word_arrs[0])
    mean_1 = sum(cluster_word_arrs[1]) / len(cluster_word_arrs[1])
    if mean_0 > mean_1:
        greater_cluster = 0
    else:
        greater_cluster = 1

    labels_agg = list(agg.labels_)

    first_occ = labels_agg.index(greater_cluster)
    last_occ = len(labels_agg) - 1 - labels_agg[::-1].index(greater_cluster)

    # Count number of headers in each chapter heading, take mode of that number
    l = [len(x.header) for x in header_contents]
    max_count = max(set(l), key=l.count)

    for idx in range(len(header_contents)):
        headers = header_contents[idx].header
        if idx >= first_occ and idx <= last_occ:
            # Add text with header tag
            start_header_index = len(headers) - max_count
            for index in range(len(headers)):
                h = headers[index]
                if index < start_header_index:
                    ans_label = 'C'
                else:
                    ans_label = 'H'
                retval.append((ans_label, h))
        else:
            # Add text with content tag
            for h in headers:
                retval.append(('C', h))

        contents = header_contents[idx].content
        for c in contents:
            retval.append(('C', c))

    return retval

def process_header_contents(header_contents):
    headers = [' '.join([elem.strip().replace('\n', ' ') for elem in x.header]) for x in header_contents]

    chapter = 0
    for w in headers:
        if w.lower().startswith('chapter'):
            chapter += 1

    part = 0
    for w in headers:
        if w.lower().startswith('part'):
            part += 1

    pattern = re.compile("^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$")
    roman = 0
    for w in headers:
        w = re.sub(r'([^\s\w]|_)+', ' ', w)
        spl = w.strip().split()
        if len(spl) > 0 and pattern.match(spl[0]):
            roman += 1

    number = 0
    for w in headers:
        spl = w.lower().replace('.', ' ').split()
        if len(spl) > 0 and spl[0].isnumeric():
            number += 1

    d = dict()
    d['chapter'] = chapter
    d['part'] = part
    d['roman'] = roman
    d['number'] = number

    descending = sorted(d.items(), key=lambda kv: -kv[1])

    if descending[0][1] == 0:
        # Nothing worked
        pass

    elif descending[0][0] == 'chapter':
        return process_chapter(header_contents)

    elif descending[0][0] == 'part':
        return process_part(header_contents)

    elif descending[0][0] == 'roman':
        if descending[0][1] < 2:
            # failed, detected 'I' pronoun as number
            pass
        else:
            return process_roman(header_contents)

    elif descending[0][0] == 'number':
        if descending[0][1] < 2:
            # failed, false positive
            pass
        else:
            return process_number(header_contents)

    return process_base_case(header_contents)


def process_book(html_dir, extracted_header_dir, book_id):
    input_location = os.path.join(html_dir, book_id + '.html')
    output_location = os.path.join(extracted_header_dir, book_id + '.csv')
    if os.path.exists(output_location):
        return book_id_str, 0
    try:
        header_content_pairs = segment_book(input_location)
        hc = process_header_contents(header_content_pairs)
        df = pd.DataFrame(hc, columns=['label', 'text'])
        df['key'] = (df['label'] != df['label'].shift(1)).astype(int).cumsum()
        df2 = pd.DataFrame(df.groupby(['key', 'label'])['text'].apply('\n\n'.join))
        df2 = df2.reset_index()[['label', 'text']]
        df2.to_csv(output_location, index=False)
        return book_id_str, 0
    except:
        return book_id_str, -1


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))
    
    # Read list of books to process
    book_list = config.get('01_Extract_headers_from_HTML', 'book_list')
    if not os.path.isfile(book_list):
        print('Please provide a valid file name for the list of book IDs in the "book_list" field.')
        exit()
    with open(book_list, 'r') as f:
        books = f.read().splitlines()
    
    # Read location of HTML files
    html_dir = config.get('01_Extract_headers_from_HTML', 'html_dir')
    if not os.path.isdir(html_dir):
        print('Please provide a valid directory name where the HTMLs are located, in the "html_dir" field.')
        exit()
    
    # Read location to store extracted headers
    extracted_header_dir = config.get('01_Extract_headers_from_HTML', 'extracted_header_dir')
    if not os.path.isdir(extracted_header_dir):
        os.makedirs(extracted_header_dir)
    
    # Read number of processes to use
    num_procs = int(config.get('01_Extract_headers_from_HTML', 'num_procs'))
    
    # Read location to store status of header extraction
    log_file = config.get('01_Extract_headers_from_HTML', 'log_file')
    
    func = partial(process_book, html_dir, extracted_header_dir)
    
    pool = multiprocessing.Pool(processes=num_procs)
    data = pool.map(func, books)
    pool.close()
    pool.join()
    
    print('Done! Saving status results to log file...')

    df = pd.DataFrame(data, columns=['bookID', 'status'])
    df.to_csv(log_file, index=False)
    
    print('Saved results to log file!')
    
