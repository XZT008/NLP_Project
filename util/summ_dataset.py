import torch
from torch.utils.data import Dataset, DataLoader
from util.data_util import doc2ids, sum2ids, word2ids
import numpy as np
from collections import namedtuple


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


class summ_dataset(Dataset):
    def __init__(self, doc, sum, word2id, MAX_ENC_LEN=100, MAX_DEC_LEN=20):

        self.doc = doc
        self.sum = sum
        self.word2id = word2id
        self.max_enc_len = MAX_ENC_LEN
        self.max_dec_len = MAX_DEC_LEN

    def __len__(self):
        return len(self.doc)

    def __getitem__(self, idx):
        if len(self.doc[idx]) == 0 or len(self.sum[idx]) == 0:
            del self.doc[idx]
            del self.sum[idx]
            return None


        doc_text_l = self.doc[idx]
        if len(doc_text_l) > self.max_enc_len:
            doc_text_l = doc_text_l[:self.max_enc_len]
        doc_len = len(doc_text_l)
        doc_input = word2ids(doc_text_l, self.word2id)

        sum_text_l = self.sum[idx]
        sum_input = word2ids(['START']+sum_text_l, self.word2id)
        sum_len = len(sum_text_l)+1

        #extend vocab for pointer generator
        doc_input_extend_vocab, doc_oovs = doc2ids(doc_text_l, self.word2id)
        sum_target = sum2ids(sum_text_l+['STOP'], self.word2id, doc_oovs)

        return np.array(doc_input), np.array(doc_input_extend_vocab), doc_len, np.array(sum_input), np.array(sum_target), \
               sum_len, doc_oovs, len(doc_oovs)


def collate_func(batch, MAX_DOC_LEN=100, MAX_SUM_LEN=30):



    doc_input = []
    doc_extend_vocab = []
    sum_input = []
    sum_target = []
    input_masks = []
    p_gen_pad = []
    doc_len = []
    sum_len = []
    oovs = []

    for datum in batch:
        doc_len.append(datum[2])
        sum_len.append(datum[5])

    MAX_LEN_DOC = np.min([np.max(doc_len), MAX_DOC_LEN])
    MAX_LEN_SUM = np.min([np.max(sum_len), MAX_SUM_LEN])

    doc_len = np.clip(doc_len, a_min=None, a_max=MAX_LEN_DOC)
    sum_len = np.clip(sum_len, a_min=None, a_max=MAX_LEN_SUM)
    # padding
    for datum in batch:
        if datum[2] > MAX_LEN_DOC:
            padded_vec_s0 = np.array(datum[0])[:MAX_LEN_DOC]
            padded_vec_s1 = np.array(datum[1])[:MAX_LEN_DOC]
        else:
            padded_vec_s0 = np.pad(np.array(datum[0]),
                                   pad_width=((0, MAX_LEN_DOC - datum[2])),
                                   mode="constant", constant_values=0)
            padded_vec_s1 = np.pad(np.array(datum[1]),
                                   pad_width=((0, MAX_LEN_DOC - datum[2])),
                                   mode="constant", constant_values=0)
        if datum[5] > MAX_LEN_SUM:
            padded_vec_s2 = np.array(datum[3])[:MAX_LEN_SUM]
            padded_vec_s3 = np.array(datum[4])[:MAX_LEN_SUM]
        else:
            padded_vec_s2 = np.pad(np.array(datum[3]),
                                   pad_width=((0, MAX_LEN_SUM - datum[5])),
                                   mode="constant", constant_values=0)
            padded_vec_s3 = np.pad(np.array(datum[4]),
                                   pad_width=((0, MAX_LEN_SUM - datum[5])),
                                   mode="constant", constant_values=0)

        mask = np.where(padded_vec_s0==0, 0, 1)

        doc_input.append(padded_vec_s0)
        doc_extend_vocab.append(padded_vec_s1)
        sum_input.append(padded_vec_s2)
        sum_target.append(padded_vec_s3)
        input_masks.append(mask)
        p_gen_pad.append(datum[7])
        oovs.append(datum[6])

    p_gen_pad_size = max(p_gen_pad)

    doc_input, doc_extend_vocab, sum_input, sum_target, input_masks,  doc_len, sum_len, oovs = argsort(doc_len, doc_input, doc_extend_vocab, sum_input,
                                                             sum_target, input_masks, doc_len, sum_len, oovs, descending=True)

    named_returntuple = namedtuple('namedtuple', ['doc_input', 'doc_extend_vocab', 'sum_input', 'sum_target', 'input_masks',
                                                  'doc_len', 'sum_len', 'vocab_pad', 'oovs'])
    return_tuple = named_returntuple(torch.from_numpy(np.array(doc_input)).long().cuda(),
                                     torch.from_numpy(np.array(doc_extend_vocab)).long().cuda(),
                                     torch.from_numpy(np.array(sum_input)).long().cuda(),
                                     torch.from_numpy(np.array(sum_target)).long().cuda(),
                                     torch.from_numpy(np.array(input_masks)).cuda(),
                                     torch.from_numpy(np.array(doc_len)),
                                     torch.from_numpy(np.array(sum_len)),
                                     torch.zeros(len(doc_len), p_gen_pad_size).cuda(),
                                     oovs)

    return return_tuple