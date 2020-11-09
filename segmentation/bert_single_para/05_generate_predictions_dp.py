import glob
import os
import pickle
import pandas as pd
import numpy as np

from functools import partial
import multiprocessing


class DPSolver():
    def __init__(self, peaks, prominences, num_sentences, breaks_to_insert, alpha):
        self.peaks = list(peaks)
        self.peaks.append(num_sentences)
        self.prominences = prominences
        self.prominences.append(0)
        self.num_sentences = num_sentences
        self.k = breaks_to_insert
        self.dp_dict = dict()
        self.N = len(self.peaks) - 1
        self.ideal_length = num_sentences / (self.k + 1)
        self.prev_dict = dict()
        self.alpha = alpha
    
    def get_distance(self, idx1, idx2):
        if idx1 == -1:
            return self.peaks[idx2] / self.ideal_length
        sent_1 = self.peaks[idx1]
        sent_2 = self.peaks[idx2]
        return (abs((sent_2 - sent_1) - self.ideal_length)) / float(self.ideal_length)
    
    def get_mins(self, costs):
        keys = list(costs.keys())
        min_key = keys[0]
        min_val = costs[keys[0]]
        for k in keys[1:]:
            if costs[k] < min_val:
                min_val = costs[k]
                min_key = k
        return min_key, min_val
    
    def dp_func(self, N, k):
        if k > N:
            return None
        # Memoized
        if (N, k) in self.dp_dict:
            return self.dp_dict[(N, k)]
        # Base case
        if k == 0:
            self.dp_dict[(N, k)] = -(self.prominences[N] * self.alpha) + (self.get_distance(-1, N) * (1 - self.alpha))
            return self.dp_dict[(N, k)]
        
        # Recursive call
        costs = dict()
        for i in range(0, N):
            c = self.dp_func(i, k - 1)
            if c:
                costs[i] = c + (self.get_distance(i, N) * (1 - self.alpha))
        if len(costs) == 0:
            self.dp_dict[(N, k)] = None
            return None
        
        min_N, min_cost = self.get_mins(costs)
        
        ans = min_cost - (self.prominences[N] * self.alpha)
        self.dp_dict[(N, k)] = ans
        self.prev_dict[(N, k)] = min_N
        return ans
    
    def solve(self):
        x = self.dp_func(self.N, self.k)
        return x
    
    def get_best_sequence(self):
        x = self.solve()
        ans_seq = list()
        N = self.N
        k = self.k
        while True:
            if (N, k) not in self.prev_dict:
                break
            previous = self.prev_dict[(N, k)]
            ans_seq.append(previous)
            N = previous
            k -= 1
        return ans_seq[::-1]
    
    
    
def get_predictions(peaks, prominences, num_preds, max_sent_num, alpha):
    
    
    dps = DPSolver(peaks, prominences, max_sent_num + 1, num_preds, alpha)
    preds = dps.get_best_sequence()
    dp_predictions = [peaks[x] for x in preds]
    return dp_predictions


def process_book(break_probs_dir, para_to_sent_dir, gt_dir, output_dir, book_id):
    with open(os.path.join(break_probs_dir, book_id + '.pkl'), 'rb') as f:
        break_probs = pickle.load(f)
    with open(os.path.join(para_to_sent_dir, book_id + '.pkl'), 'rb') as f:
        para_to_sent = pickle.load(f)
    
    peaks = list()
    prominences = list()
    for n, prob in break_probs.items():
        if prob > 0.9:
            peaks.append(para_to_sent[n])
            prominences.append(np.log(prob))
    
    with open(os.path.join(gt_dir, book_id + '_gt_sents.pkl'), 'rb') as f:
        gt = pickle.load(f)
    with open(os.path.join(gt_dir, book_id + '_max_sent_num.pkl'), 'rb') as f:
        max_sent_num = int(pickle.load(f))
        
    num_preds = len(gt)
    
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        preds = get_predictions(peaks, prominences, num_preds, max_sent_num, alpha)
        with open(os.path.join(output_dir, book_id + '_alpha_' + str(int(alpha * 100)) + '.pkl'), 'wb') as f:
            pickle.dump(preds, f)
    
    print(book_id, 'success')
    return book_id, 'Success'
    
    
    
    
if __name__ == "__main__":
    
    # Use appropriate locations
    test_books_list_file = 'test_books.txt'
    with open(test_books_list_file, 'r') as f:
        test_book_ids = [x.strip() for x in f.readlines()]

    gt_dir = 'test_gt_sentences/'
    
    break_probs_dir = 'test_preds/'
    
    para_to_sent_dir = 'test_books_para_to_sent/'
    
    output_dir = 'thresh_0_9/'
    
    if not os.path.exists(break_probs_dir):
        print('Invalid break probs dir')
        exit()
    
    if not os.path.exists(gt_dir):
        print('Invalid ground truth dir')
        exit()
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    func = partial(process_book, break_probs_dir, para_to_sent_dir, gt_dir, output_dir)
    
    pool = multiprocessing.Pool(processes=32)
    data = pool.map(func, test_book_ids)
    pool.close()
    pool.join()
    
    df = pd.DataFrame(data)
    df.rename(columns={0:'book_id', 1:'status'}, inplace=True)
    df.to_csv('./log_file_dp_single_para.csv', index=False)
    print('Done!')
    
