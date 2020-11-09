from nltk.metrics import segmentation as seg
import pickle
import pandas as pd
import os

import multiprocessing
from functools import partial

def get_standard_metrics(gt, pred, msn):
    gt_segs = ''.join(['1' if i in gt else '0' for i in range(msn)])
    pred_segs = ''.join(['1' if i in pred else '0' for i in range(msn)])
    k_val = int(round(len(gt_segs) / (gt_segs.count('1') * 2.0)))
    k_val = k_val // 4
    return seg.pk(gt_segs, pred_segs, k=k_val), seg.windowdiff(gt_segs, pred_segs, k=k_val)

def get_prec_rec_f1(gt, pred):
    tp = len([x for x in pred if x in gt])
    fp = len([x for x in pred if x not in gt])
    fn = len([x for x in gt if x not in pred])
    
    precision, recall, f1 = None, None, None
    
    try:
        precision = tp / (tp + fp)
    except:
        pass
    try:
        recall = tp / (tp + fn)
    except:
        pass
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        pass
    
    return precision, recall, f1


def get_gt_msn(gt_dir, book_id):
    with open(os.path.join(gt_dir, book_id + '_gt_sents.pkl'), 'rb') as f:
        gt = pickle.load(f)
    with open(os.path.join(gt_dir, book_id + '_max_sent_num.pkl'), 'rb') as f:
        max_sent_num = int(pickle.load(f))
    return gt, max_sent_num

def get_pred(pred_dir, book_id, alpha):
    with open(os.path.join(pred_dir, book_id + '_alpha_' + str(alpha) + '.pkl'), 'rb') as f:
        preds = pickle.load(f)
    return preds

def get_metrics(gt_loc, pred_loc, book_id):
    gt, msn = get_gt_msn(gt_loc, book_id)
    
    ans = list()
    
    for alpha in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        pk, wd, p, r, f1 = None, None, None, None, None

        try:
            pred = get_pred(pred_loc, book_id, alpha)
        except:
            ans.append([book_id, pk, wd, p, r, f1])
            continue

        try:
            pk, wd = get_standard_metrics(gt, pred, msn)
        except:
            pass
        try:
            p, r, f1 = get_prec_rec_f1(gt, pred)
        except:
            pass
        ans.append([book_id, pk, wd, p, r, f1])
    return ans


if __name__ == "__main__":
    
    # Use appropriate locations
    gt_loc = 'test_gt_sentences/'
    
    pred_loc = 'woc/window_size_200/dp/'
    output_dir = 'woc/window_size_200/results/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_books_list_file = 'test_books.txt'
    with open(test_books_list_file, 'r') as f:
        test_book_ids = [x.strip() for x in f.readlines()]
        
    d = dict()
    for idx in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        d[idx] = list()

    func = partial(get_metrics, gt_loc, pred_loc)
    
    pool = multiprocessing.Pool(processes=32)
    data = pool.map(func, test_book_ids)
    pool.close()
    pool.join()
    
    
    for metrics_b in data:
        for idx, elem in enumerate(metrics_b):
            d[idx * 10].append(elem)
    
    for alpha in d:
        print(alpha)
        df = pd.DataFrame(d[alpha], columns=['book_id', 'pk', 'wd', 'precision', 'recall', 'f1'])
        df.to_csv(os.path.join(output_dir, 'alpha_' + str(alpha) + '.csv'))
        
    print('Done!')

