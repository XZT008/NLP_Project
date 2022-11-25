from functools import partial
from torch.utils.data import Dataset, DataLoader
import torch
from util.summ_dataset import summ_dataset, collate_func
import pickle
from tqdm import tqdm
from util import config
from beam_search import beam_search
from rouge import Rouge

model = torch.load('saved/saved_model/rl_1.pt')
with open('saved/test_document.pkl', 'rb') as f:
    test_doc = pickle.load(f)

with open('saved/test_summary.pkl', 'rb') as f:
    test_sum = pickle.load(f)

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


def test(model, id2word, dataloader):
    model.eval()
    inputs = []
    preds = []
    targets = []
    rouge = Rouge()
    for i, batch in tqdm(enumerate(dataloader)):
        doc_input, doc_extend_vocab, sum_input, sum_target, input_masks, doc_len, sum_len, vocab_pad, oovs = \
            batch.doc_input, batch.doc_extend_vocab, batch.sum_input, batch.sum_target, \
            batch.input_masks, batch.doc_len, batch.sum_len, batch.vocab_pad, batch.oovs

        with torch.no_grad():
            enc_batch = model.embeds(doc_input)
            enc_out, enc_hidden = model.encoder(enc_batch, doc_len)
            ct_e = torch.zeros(doc_len.size(0), 2 * config.hidden_dim).cuda()
            pred_ids = beam_search(enc_hidden, enc_out, input_masks, ct_e, vocab_pad, doc_extend_vocab,
                                   model, 2, 3, 1)

            for doc_input, pred, target, oov in zip(doc_input, pred_ids, sum_target, oovs):
                tmp_vocab = id2word+oov
                inputs.append(' '.join([tmp_vocab[i] for i in doc_input if i != 0 and i != 3]))
                preds.append(' '.join([tmp_vocab[i] for i in pred]))
                targets.append(' '.join([tmp_vocab[i] for i in target if i != 0 and i != 3]))


            scores = rouge.get_scores(preds, targets, avg=True)
            print("Scores: ", scores)



test_doc, test_sum = cleaning(test_doc, test_sum)
test_dataset = summ_dataset(test_doc, test_sum, word2id, MAX_ENC_LEN=200, MAX_DEC_LEN=30)
test_loader = DataLoader(test_dataset, batch_size=200, collate_fn=partial(collate_func, MAX_DOC_LEN=200, MAX_SUM_LEN=31)
                          , shuffle=True)
test(model, id2word, test_loader)
