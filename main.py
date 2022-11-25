import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from util.data_util import Vocab
import pickle
from util.summ_dataset import summ_dataset
from functools import partial
from util.summ_dataset import collate_func
from train import train_loop
from model import Model
from tqdm import tqdm
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

with open('saved/id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)

def cleaning(docs, sums):
    idx_list = []
    for i,(d, s) in tqdm(enumerate(zip(docs, sums))):
        if len(d) == 0 or len(s) == 0:
            idx_list.append(i)
    for i in sorted(idx_list, reverse=True):
        del docs[i]
        del sums[i]
    return docs, sums


train_doc, train_sum = cleaning(train_doc, train_sum)


train_dataset = summ_dataset(train_doc, train_sum, word2id, MAX_ENC_LEN=200, MAX_DEC_LEN=30)


train_loader = DataLoader(train_dataset, batch_size=200, collate_fn=partial(collate_func, MAX_DOC_LEN=200, MAX_SUM_LEN=31)
                          , shuffle=True)

#model = Model()
model = torch.load('saved/saved_model/rl_1.pt')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loop(20, train_loader, model, optimizer, id2word)

