import torch
from torch.utils.data import Dataset, DataLoader
from util.data_util import words2ids
import numpy as np
from collections import namedtuple

class summ_dataset(Dataset):
    def __init__(self, doc, sum, word2id):
        self.doc = doc
        self.sum = sum
        self.word2id = word2id

    def __len__(self):
        return len(self.doc)

    def __getitem__(self, idx):
        doc_text_l = self.doc[idx]
        sum_text_l = ['START'] + self.sum[idx] + ['STOP']
        doc_tensor = np.array(words2ids(doc_text_l, self.word2id))
        sum_tensor = np.array(words2ids(sum_text_l, self.word2id))
        return doc_tensor, len(doc_tensor), sum_tensor, len(sum_tensor)


def argsort(keys, *lists, descending=False):
    """Reorder each list in lists by the (descending) sorted order of keys.
    :param iter keys: Keys to order by.
    :param list[list] lists: Lists to reordered by keys's order.
                             Correctly handles lists and 1-D tensors.
    :param bool descending: Use descending order if true.
    :returns: The reordered items.
    """
    ind_sorted = sorted(range(len(keys)), key=lambda k: keys[k])
    if descending:
        ind_sorted = list(reversed(ind_sorted))
    output = []
    for lst in lists:
        if isinstance(lst, torch.Tensor):
            output.append(lst[ind_sorted])
        else:
            output.append([lst[i] for i in ind_sorted])
    return output


def collate_func(batch, MAX_DOC_LEN=100, MAX_SUM_LEN=30):
    doc_data = []
    sum_data = []
    doc_len = []
    sum_len = []

    for datum in batch:
        doc_len.append(datum[1])
        sum_len.append(datum[3])

    MAX_LEN_DOC = np.min([np.max(doc_len), MAX_DOC_LEN])
    MAX_LEN_SUM = np.min([np.max(sum_len), MAX_SUM_LEN])

    doc_len = np.clip(doc_len, a_min=None, a_max=MAX_LEN_DOC)
    sum_len = np.clip(sum_len, a_min=None, a_max=MAX_LEN_SUM)
    # padding
    for datum in batch:
        if datum[1] > MAX_LEN_DOC:
            padded_vec_s1 = np.array(datum[0])[:MAX_LEN_DOC]
        else:
            padded_vec_s1 = np.pad(np.array(datum[0]),
                                   pad_width=((0, MAX_LEN_DOC - datum[1])),
                                   mode="constant", constant_values=0)
        if datum[3] > MAX_LEN_SUM:
            padded_vec_s2 = np.array(datum[2])[:MAX_LEN_SUM]
        else:
            padded_vec_s2 = np.pad(np.array(datum[2]),
                                   pad_width=((0, MAX_LEN_SUM - datum[3])),
                                   mode="constant", constant_values=0)
        doc_data.append(padded_vec_s1)
        sum_data.append(padded_vec_s2)
    """
    packed = True
    if packed:
        source_data, source_len, target_data, target_len = argsort(doc_len, doc_data, doc_len, sum_data,
                                                                   sum_len, descending=True)
    """

    named_returntuple = namedtuple('namedtuple', ['doc_vecs', 'doc_lens', 'sum_vecs', 'sum_lens'])
    return_tuple = named_returntuple(torch.from_numpy(np.array(doc_data)),
                                     torch.from_numpy(np.array(doc_len)),
                                     torch.from_numpy(np.array(sum_data)),
                                     torch.from_numpy(np.array(sum_len)));

    return return_tuple