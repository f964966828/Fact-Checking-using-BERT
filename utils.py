import os
import json
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertTokenizer, BertModel

class_num = 3
article_num = 12
batch_size = 1
sequence_length = 512

def text_preprocessing(string):
    string = string.replace('-', '')
    string = string.replace('_', '')
    string = string.replace('  ', '')
    string = string.replace('...', '')
    return string

def get_dataloader(all_data, all_evidence, shuffle=True, mode="train"):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print(f"Loading {mode} data")
    
    all_input_ids = list()
    all_input_mask = list()
    all_segment_ids = list()
    all_label_ids = list()
    for data in tqdm(all_data):
        claim = data['metadata']['claim']
        file_id = str(data['metadata']['id'])

        if mode != "test":
            label = data['label']['rating']
        
        sentences = list()
        for evidence in all_evidence[file_id]:
            sentence = text_preprocessing(evidence)
            sentences.append(sentence)
            
            if len(sentences) >= article_num:
                break           

        encoding = tokenizer(
            text = [claim] * article_num, 
            text_pair = sentences + [""] * (article_num - len(sentences)),
            return_tensors = "pt",
            padding = 'max_length',
            truncation = 'only_second',
            max_length = sequence_length
        )

        all_input_ids.append(encoding.input_ids.unsqueeze(0))
        all_input_mask.append(encoding.attention_mask.unsqueeze(0))
        all_segment_ids.append(encoding.token_type_ids.unsqueeze(0))
        
        if mode != "test":
            all_label_ids.append(torch.tensor(label).unsqueeze(0))
            
    if mode == "train":
        print("Loading augment Data")

        with open('./2022-inlp-final/aug_evidence.json') as f:
            aug_data_evidence = json.load(f)

        for idx in tqdm(aug_data_evidence):
            data = aug_data_evidence[idx]
            
            claim = data['claim']
            label = data['label']
            
            sentences = list()
            for evidence in data['sentences']:
                sentence = text_preprocessing(evidence)
                sentences.append(sentence)
                
                if len(sentences) >= article_num:
                    break           
        

            encoding = tokenizer(
                text = [claim] * article_num, 
                text_pair = sentences + [""] * (article_num - len(sentences)),
                return_tensors = "pt",
                padding = 'max_length',
                truncation = 'only_second',
                max_length = sequence_length
            )

            all_input_ids.append(encoding.input_ids.unsqueeze(0))
            all_input_mask.append(encoding.attention_mask.unsqueeze(0))
            all_segment_ids.append(encoding.token_type_ids.unsqueeze(0))
            all_label_ids.append(torch.tensor(label).unsqueeze(0))

    all_input_ids = torch.cat(all_input_ids)
    all_input_mask = torch.cat(all_input_mask)
    all_segment_ids = torch.cat(all_segment_ids)
    
    if mode != "test":
        all_label_ids = torch.cat(all_label_ids)
    
    if mode != "test":
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader

class BertForFactChecking(nn.Module):
    def __init__(self):
        super(BertForFactChecking, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(768, class_num)

    def forward(self, input_ids, attention_mask, token_type_ids):
        
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, article_num, class_num)
            
        return reshaped_logits.sum(1)
