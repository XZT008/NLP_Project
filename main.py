import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from util.data_util import Vocab
import pickle
from util.summ_dataset import summ_dataset
from functools import partial
from util.summ_dataset import collate_func
"""
This section is to tokenize data


dataset = load_dataset("xsum")
data_dict = {
    "train_document" : dataset['train']['document'],
    "train_summary" : dataset['train']['summary'],
    "valid_document" : dataset['validation']['document'],
    "valid_summary" : dataset['validation']['summary'],
    "test_document" : dataset['test']['document'],
    "test_summary" : dataset['test']['summary'],
}
vocab = Vocab()
vocab.tokenize_list_of_string(data_dict, 'saved')
"""

"""
This section is to build Vocab


vocab = Vocab()
with open('saved/train_document.pkl', 'rb') as f:
    train_doc = pickle.load(f)

with open('saved/train_summary.pkl', 'rb') as f:
    train_sum = pickle.load(f)

train_data = train_doc+train_sum
vocab.build_vocab(train_data, 50000)
"""


with open('saved/train_document.pkl', 'rb') as f:
    train_doc = pickle.load(f)

with open('saved/train_summary.pkl', 'rb') as f:
    train_sum = pickle.load(f)

with open('saved/word2id.pkl', 'rb') as f:
    word2id = pickle.load(f)

train_dataset = summ_dataset(train_doc, train_sum, word2id)
train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=partial(collate_func, MAX_DOC_LEN=200, MAX_SUM_LEN=31))
for t in train_loader:
    break                   #for debug
