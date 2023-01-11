import os
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import get_dataloader, BertForFactChecking
from transformers import logging

num_epoch = 3
show_freq = 10
DEVICE = 'cuda'

with open('./2022-inlp-final/train.json') as f:
    train_data = json.load(f)
with open('./2022-inlp-final/valid.json') as f:
    valid_data = json.load(f)
with open('./2022-inlp-final/train_evidence.json') as f:
    train_data_evidence = json.load(f)
with open('./2022-inlp-final/valid_evidence.json') as f:
    valid_data_evidence = json.load(f)
with open('./2022-inlp-final/aug_evidence.json') as f:
    aug_data_evidence = json.load(f)

print(f"Training Dataset Size: {len(train_data) + len(aug_data_evidence)}")
print(f"Validation Dataset Size: {len(valid_data)}")

logging.set_verbosity_error()
model = BertForFactChecking()
model = model.to(DEVICE)
model.load_state_dict(torch.load('bert_weight.pth'))

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

train_loader = get_dataloader(train_data, train_data_evidence, mode="train")
valid_loader = get_dataloader(valid_data, valid_data_evidence, mode="valid")

best_valid_f1_score = 0.0
for epoch in range(num_epoch):
    
    epoch_start_time = time.time()
    train_loss, valid_loss = 0.0, 0.0
    train_count, valid_count = 0.0, 0.0
    train_true, valid_true = list(), list()
    train_pred, valid_pred = list(), list()
    
    model.train()
    for i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(train_loader):
        input_ids = input_ids.to(DEVICE)
        input_mask = input_mask.to(DEVICE)
        segment_ids = segment_ids.to(DEVICE)
        label_ids = label_ids.to(DEVICE)
        
        outputs = model(
            input_ids=input_ids, 
            token_type_ids=segment_ids, 
            attention_mask=input_mask,
        )
        
        batch_loss = loss(outputs, label_ids)
        
        batch_loss.backward()
        optimizer.step()      
        model.zero_grad()
                
        train_loss += batch_loss.item()
        train_count += input_ids.shape[0]
        train_true += label_ids.tolist()
        train_pred += np.argmax(outputs.cpu().detach().numpy(), axis=1).tolist()
        
        if (i+1) % show_freq == 0 or (i+1) == len(train_loader):
            train_acc = accuracy_score(train_true, train_pred)
            train_f1_score = f1_score(train_true, train_pred, average='macro')
            print('[{:02d}/{:02d} - {:04d}/{:04d}] '.format(epoch+1, num_epoch, i+1, len(train_loader))
                + '{:2.2f} sec '.format(time.time() - epoch_start_time)
                + 'Train Acc: {:3.2f}% Loss: {:3.4f} '.format(train_acc*100, train_loss/train_count)
                + 'F1 Score: {:3.4f}'.format(train_f1_score)
            )
            
    model.eval()    
    for i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(valid_loader):
        input_ids = input_ids.to(DEVICE)
        input_mask = input_mask.to(DEVICE)
        segment_ids = segment_ids.to(DEVICE)
        label_ids = label_ids.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask,
            )
            
        batch_loss = loss(outputs, label_ids)
        
        valid_loss += batch_loss.item()
        valid_count += input_ids.shape[0]
        valid_true += label_ids.tolist()
        valid_pred += np.argmax(outputs.cpu().detach().numpy(), axis=1).tolist()
        
        if (i+1) % show_freq == 0 or (i+1) == len(valid_loader):
            valid_acc = accuracy_score(valid_true, valid_pred)
            valid_f1_score = f1_score(valid_true, valid_pred, average='macro')
            print('[{:02d}/{:02d} - {:04d}/{:04d}] '.format(epoch+1, num_epoch, i+1, len(valid_loader))
                + '{:2.2f} sec '.format(time.time() - epoch_start_time)
                + 'Valid Acc: {:3.2f}% Loss: {:3.4f} '.format(valid_acc*100, valid_loss/valid_count)
                + 'F1 Score: {:3.4f}'.format(valid_f1_score)
            )
            
    if best_valid_f1_score < valid_f1_score:
        best_valid_f1_score = valid_f1_score
        torch.save(model.state_dict(), 'bert_weight.pth')
        
print(f"Best Validation F1 Score: {best_valid_f1_score}")
