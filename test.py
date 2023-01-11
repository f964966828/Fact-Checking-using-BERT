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

class_num = 3
article_num = 5
batch_size = 8
sequence_length = 200

num_epoch = 3
show_freq = 10
DEVICE = 'cuda'

with open('./2022-inlp-final/test.json') as f:
    test_data = json.load(f)
with open('./2022-inlp-final/test_evidence.json') as f:
    test_data_evidence = json.load(f)

print(f"Testing Dataset Size: {len(test_data)}")

logging.set_verbosity_error()
model = BertForFactChecking()
model = model.to(DEVICE)
model.load_state_dict(torch.load('bert_weight.pth'))
         
test_pred = list()
for (input_ids, input_mask, segment_ids) in tqdm(test_loader):
    input_ids = input_ids.to(DEVICE)
    input_mask = input_mask.to(DEVICE)
    segment_ids = segment_ids.to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, 
            token_type_ids=segment_ids, 
            attention_mask=input_mask,
        )

    test_pred += np.argmax(outputs.cpu().detach().numpy(), axis=1).tolist()

df = pd.DataFrame({
    'id': [data['metadata']['id'] for data in test_data],
    'rating': test_pred
})

df.to_csv('submission.csv', index=False)
