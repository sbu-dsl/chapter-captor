import argparse
import configparser
import os
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForTokenClassification
from tqdm import tqdm, trange
import numpy as np
import pickle


# Accuracy metrics function
def flat_accuracy(seq_pred, seq_labels):
    m = seq_pred.argmax(axis=2)
    m2 = m.flatten()
    m2 = m2.detach().cpu().numpy()
    l2 = seq_labels.flatten()
    l2 = l2.to('cpu').numpy()
    tp, tn, fp, fn = 0, 0, 0, 0
    for idx in range(len(m2)):
        if l2[idx] == 1 and m2[idx] == 1:
            tp += 1
        elif l2[idx] == 1 and m2[idx] == 0:
            fn += 1
        elif l2[idx] == 0 and m2[idx] == 1:
            fp += 1
        elif l2[idx] == 0 and m2[idx] == 0:
            tn += 1
    return np.sum(m2 == l2) / len(l2), tp, tn, fp, fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Configuration file', required=True)
    args = parser.parse_args()
    config_file = args.config_file

    config = configparser.ConfigParser()
    config.read_file(open(config_file))
    
    # Read location to store model checkpoints
    checkpoint_dir = config.get('03_Train_model', 'checkpoint_dir')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Read set of training books
    # Read the list of book IDs in the training set
    train_set_books_file = config.get('02_Generate_training_sequences', 'train_books_list')
    if not os.path.isfile(train_set_books_file):
        print('Please provide a valid file name for the list of training set book IDs in the "train_books_list" field.')
        exit()
    with open(train_set_books_file) as f:
        train_book_ids = [x.strip() for x in f.readlines()]
        
    # Read the number of epochs to train for
    num_epochs = int(config.get('03_Train_model', 'num_epochs'))
    
    
    # Read the directory location where generated sequences are stored
    seq_gen_dir = config.get('02_Generate_training_sequences', 'generated_sequence_dir')
    if not os.path.isdir(seq_gen_dir):
        print('Please run 02_generate_training_sequences.py first.')
        exit()
    
    
    # Read token and label sequences
    token_list = list()
    label_list = list()

    for book_index in train_book_ids:
        try:
            with open(os.path.join(seq_gen_dir, book_index + '_tokens.pkl'), 'rb') as f:
                t_list = pickle.load(f)
            with open(os.path.join(seq_gen_dir, book_index + '_labels.pkl'), 'rb') as f:
                l_list = pickle.load(f)
            token_list += t_list
            label_list += l_list
        except Exception as e:
            print('Could not fetch sequences for: ' + book_index)
            print(e)
            continue
    
    print(len(token_list), " sequences in training data")
    
    
    # Train-validation split for loaded data
    print("Train-validation split")
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(token_list, label_list, random_state=2019, test_size=0.1)
    
    
    # Converting data to tensor form
    print("Initializing tensors")
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    # Batch size
    batch_size = 32
    
    
    # Creating objects to use in training
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    
    # GPU / CPU initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)
    
    
    
    # Model to fine-tune
    print("Initializing model")
    model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=2)

    
    
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
    
    
    # Training loop
    train_loss_set = []
    epochs = num_epochs

    count = 1
    
    
    for _ in trange(epochs, desc='Epoch'):
        print("Epoch " + str(count))
        # Set to training mode
        model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Process each batch
        for step, batch in enumerate(train_dataloader):
            # Convert batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack
            b_input_ids, b_labels = batch
            # Reset gradients to 0
            optimizer.zero_grad()
            # Compute loss
            loss = model(b_input_ids, labels=b_labels)
            # Append loss to list
            train_loss_set.append(loss.item())
            # Back-prop
            loss.backward()
            # Update optimizer params
            optimizer.step()
            # Add to total training loss
            tr_loss += loss.item()
            # Add to number of training examples
            nb_tr_examples += b_input_ids.size(0)
            # Add to number of training steps
            nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss/nb_tr_steps))


        # Set to eval mode
        model.eval()

        eval_loss, eval_accuracy, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Process each batch
        for batch in validation_dataloader:
            # Convert batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack
            b_input_ids, b_labels = batch
            # Do not update gradients
            with torch.no_grad():
                # Get logits
                logits = model(b_input_ids, token_type_ids=None)
            # Compute metrics
            tmp_eval_accuracy, tmp_tp, tmp_tn, tmp_fp, tmp_fn = flat_accuracy(logits, b_labels)
            eval_accuracy += tmp_eval_accuracy
            tp += tmp_tp
            tn += tmp_tn
            fp += tmp_fp
            fn += tmp_fn
            nb_eval_steps += 1


        print('Validation Accuracy: {}'.format(eval_accuracy/nb_eval_steps))
        print("TP = ", tp)
        print("TN = ", tn)
        print("FP = ", fp)
        print("FN = ", fn)
        print("----------------")

        # Save model
        torch.save({
            'epoch': count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, os.path.join(checkpoint_dir, 'epoch_' + str(count) + '.pt'))

        count += 1
    
    print('Done!')
    
    