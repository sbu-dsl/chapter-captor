import pandas as pd
import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert import BertAdam
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences

import time
import datetime

import pickle

from ast import literal_eval

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_tokens(tokenizer, toks1, toks2, cls, sep):
    toks1 = [cls] + toks1 + [sep]
    toks2 = toks2 + [sep]
    
    indexed_tokens = toks1 + toks2
    segments_ids = [0] * len(toks1) + [1] * len(toks2)
    
    return indexed_tokens, segments_ids

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    
    # Use appropriate locations
    training_data_loc = 'train_sequences_tokenized.csv'
    output_loc = 'trained_model/'
    
    print('Reading training data file...')
    
    df = pd.read_csv(training_data_loc, usecols=['para1_tokens', 'para2_tokens', 'para1_len', 'para2_len', 'label'])
    
    df = df[(df['para1_len'] > 0) & (df['para2_len'] > 0)]
    
    df['para1_tokens'] = df['para1_tokens'].apply(literal_eval)
    df['para2_tokens'] = df['para2_tokens'].apply(literal_eval)
    
    
    print('Loading tokenizer and BertNSP...')
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    BertNSP=BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    
    
    
    cls, sep = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
    
    # Use appropriate locations
    try:
        with open(os.path.join(output_loc, 'input_tokens.pkl'), 'rb') as f:
            input_tokens = pickle.load(f)
        with open(os.path.join(output_loc, 'input_seg_ids.pkl'), 'rb') as f:
            input_seg_ids = pickle.load(f)
        with open(os.path.join(output_loc, 'labels.pkl'), 'rb') as f:
            labels = pickle.load(f)
    
    except:
        print('Generating training input...')
        input_tokens = list()
        input_seg_ids = list()
        labels = list()
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(idx)
            indexed_tokens, segments_ids = get_tokens(tokenizer, row['para1_tokens'], row['para2_tokens'], cls, sep)
            input_tokens.append(indexed_tokens)
            input_seg_ids.append(segments_ids)
            labels.append(row['label'])
        # Use appropriate locations
        with open(os.path.join(output_loc, 'input_tokens.pkl'), 'wb') as f:
            pickle.dump(input_tokens, f)
        with open(os.path.join(output_loc, 'input_seg_ids.pkl'), 'wb') as f:
            pickle.dump(input_seg_ids, f)
        with open(os.path.join(output_loc, 'labels.pkl'), 'wb') as f:
            pickle.dump(labels, f)
    
    input_ids = pad_sequences(input_tokens, maxlen=512, dtype="long", value=0, truncating="pre", padding="post")
    seg_ids = pad_sequences(input_seg_ids, maxlen=512, dtype="long", value=1, truncating="pre", padding="post")
    attention_masks = [[int(token_id > 0) for token_id in sent] for sent in input_ids]
    
    
    train_input_ids, validation_input_ids, train_seg_ids, validation_seg_ids, train_attention_masks, validation_attention_masks, train_labels, validation_labels = train_test_split(input_ids, seg_ids, attention_masks, labels, random_state=2019, test_size=0.1)
    
    
    train_input_ids = torch.tensor(train_input_ids)
    validation_input_ids = torch.tensor(validation_input_ids)
    train_seg_ids = torch.tensor(train_seg_ids)
    validation_seg_ids = torch.tensor(validation_seg_ids)
    train_attention_masks = torch.tensor(train_attention_masks)
    validation_attention_masks = torch.tensor(validation_attention_masks)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    
    
    batch_size = 32
    
    
    train_data = TensorDataset(train_input_ids, train_seg_ids, train_attention_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    
    validation_data = TensorDataset(validation_input_ids, validation_seg_ids, validation_attention_masks, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    
    print("Initializing GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)
    
    
    model = BertNSP
    
    
    # Parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
    ]

    # Optimizer
    optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

    print("Making model GPU compatible")
    model = model.to(device)

    
    epochs = 4
    
    print('Starting training...')
    
    loss_values = []
    for epoch_i in range(0, epochs):
        print('Epoch ', epoch_i)
        
        model.train()
        
        t0 = time.time()
        
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_seg_ids, b_attention_masks, b_labels = batch

            optimizer.zero_grad()

            outputs = model(b_input_ids, token_type_ids=b_seg_ids, attention_mask=b_attention_masks, next_sentence_label=b_labels)

            loss = outputs[0]

            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        model.eval()
        t0 = time.time()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_seg_ids, b_attention_masks, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=b_seg_ids, attention_mask=b_attention_masks)

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        
        output_dir = output_loc + 'model_' + str(epoch_i)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print('Saved model to ' + output_dir)
