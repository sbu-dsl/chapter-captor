import os
import glob
import pickle
import lxml.etree as ET
import pandas as pd
from scipy import signal

from functools import partial
import multiprocessing


def process_book(para_to_sent_dir, density_dir, output_dir, book_id):
    
    
    # Read densities file
    with open(os.path.join(density_dir, book_id + '.pkl'), 'rb') as f:
        densities = pickle.load(f)
    
    # Read para_to_sent file
    with open(os.path.join(para_to_sent_dir, book_id + '.pkl'), 'rb') as f:
        para_to_sent = pickle.load(f)
    
    # Get valid sentence numbers (that come at ends of paragraphs)
    valid_sent_nums = list(para_to_sent.values())
    
    # Corresponding densities
    valid_densities = [densities[x] for x in sorted(valid_sent_nums[:-1])]
    
    # Get peak indices and prominences
    peaks, _ = signal.find_peaks([-x for x in valid_densities])
    prominences = signal.peak_prominences([-x for x in valid_densities], peaks)[0]
    
    # Get sentence numbers corresponding to peak indices
    peak_sent_nums = [valid_sent_nums[idx] for idx in peaks]
    peak_sent_proms = prominences
    
    with open(os.path.join(output_dir, book_id + '.pkl'), 'wb') as f:
        pickle.dump([list(peak_sent_nums), list(peak_sent_proms)], f)
    
    print(book_id, ' success!')
    return book_id, 'Success'



if __name__ == "__main__":
    para_to_sent_dir = 'test_books_para_to_sent/'
    density_dir = 'window_size_200/test_books_densities/'
    
    output_dir = 'window_size_200/test_books_peaks_proms/'
    
    if not os.path.exists(density_dir):
        print('Invalid lemma dir')
        exit()
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_books_list_file = 'test_books.txt'
    with open(test_books_list_file, 'r') as f:
        test_book_ids = [x.strip() for x in f.readlines()]
    
    print(len(test_book_ids), 'books')
    
    func = partial(process_book, para_to_sent_dir, density_dir, output_dir)
    
    pool = multiprocessing.Pool(processes=32)
    data = pool.map(func, test_book_ids)
    pool.close()
    pool.join()
    
    df = pd.DataFrame(data)
    df.rename(columns={0:'book_id', 1:'status'}, inplace=True)
    df.to_csv('./log_file_peaks.csv', index=False)
    print('Done!')
