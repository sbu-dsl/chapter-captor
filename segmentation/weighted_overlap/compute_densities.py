import glob
import os
import pickle
from functools import partial
import multiprocessing
import pandas as pd
from nltk.corpus import stopwords

def build_graph(lemma_dict, N):
    edges = list()
    for idx in range(max(lemma_dict.keys()) + 1):
        for n in range(1, 1 + N):
            if idx + n not in lemma_dict:
                continue
            common_lemmas = lemma_dict[idx].intersection(lemma_dict[idx + n])
            new_common_lemmas = set()
            for x in common_lemmas:
                if x not in "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~":
                    new_common_lemmas.add(x)
            common_lemmas = new_common_lemmas
            for i in range(len(common_lemmas)):
                edges.append([idx, idx + n])
    return edges

def compute_density(edges, max_sent_num):
    density = {x: 0 for x in range(0, max_sent_num)}
    for x, y in edges:
        for i in range(x, y):
            left_dist = i - x + 1
            right_dist = y - i
            density[i] += 1 / (left_dist + right_dist)
    return dict(density)

def process_book(lemma_dir, output_dir, N, book_id):
    with open(os.path.join(lemma_dir, book_id + '.pkl'), 'rb') as f:
        lemmas = pickle.load(f)
    
    for k in lemmas:
        lemmas[k] = set(lemmas[k])
        lemmas[k] = lemmas[k].difference(stop_words)
    
    edges = build_graph(lemmas, N)
    
    density = compute_density(edges, max(lemmas.keys()))
    # Save density to pkl file
    with open(os.path.join(output_dir, book_id + '.pkl'), 'wb') as f:
        pickle.dump(density, f)
    print(book_id, 'success')
    return book_id, 'Success'


if __name__ == "__main__":
    lemma_dir = 'use_books_lemmas/'
    output_dir = 'window_size_200/test_books_densities/'
    if not os.path.exists(lemma_dir):
        print('Invalid lemma dir')
        exit()
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_books_list_file = 'test_books.txt'
    with open(test_books_list_file, 'r') as f:
        test_book_ids = [x.strip() for x in f.readlines()]
    
    print(len(test_book_ids), 'books')
    
    stop_words = set(stopwords.words('english'))
    
    func = partial(process_book, lemma_dir, output_dir, 200)
    
    pool = multiprocessing.Pool(processes=48)
    data = pool.map(func, test_book_ids)
    pool.close()
    pool.join()
    
    df = pd.DataFrame(data)
    df.rename(columns={0:'book_id', 1:'status'}, inplace=True)
    df.to_csv('./log_file.csv', index=False)
    print('Done!')
