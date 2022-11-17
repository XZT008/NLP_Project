import spacy
from spacy.lang.en import English
from tqdm import tqdm
import pickle
from collections import Counter


def words2ids(words, word2id):
    ret = []
    for word in words:
        ret.append(word2id.get(word, 1))
    return ret


def ids2words(ids, id2word):
    ret = []
    for id in ids:
        ret.append(id2word[id])
    return ret


class Vocab(object):

    def __init__(self):
        self.tokenizer = English().tokenizer
        self.word2id = {}
        self.id2word = []
        self.PAD_TOKEN = 'PAD'              #0
        self.UNKNOWN_TOKEN = 'UNK'          #1
        self.START_DECODING = 'START'       #2
        self.STOP_DECODING = 'STOP'         #3
        self.count = 0

    def tokenize_list_of_string(self, data_dict, save_dir='saved'):
        for key, value in data_dict.items():
            tokenized_list = []
            for s in tqdm(value):
                tokens = self.tokenizer(s)
                tokenized_list.append([token.text for token in tokens])
            with open(save_dir + f'/{key}.pkl', 'wb') as f:
                pickle.dump(tokenized_list, f)

    def build_vocab(self, tokenized_data, max_vocab=50000, word2id=None, id2word=None):
        if word2id is not None and id2word is not None:
            self.word2id = word2id
            self.id2word = id2word
            print("Vocab loaded from provided data!")
            return

        all_tokens = [token for tokens in tokenized_data for token in tokens]
        token_counter = Counter(all_tokens)
        vocab, count = zip(*token_counter.most_common(max_vocab))
        self.word2id = dict(zip(vocab, range(4, 4 + len(vocab))))
        self.word2id[self.PAD_TOKEN] = 0
        self.word2id[self.UNKNOWN_TOKEN] = 1
        self.word2id[self.START_DECODING] = 2
        self.word2id[self.STOP_DECODING] = 3

        self.id2word = [self.PAD_TOKEN, self.UNKNOWN_TOKEN, self.START_DECODING, self.STOP_DECODING] + list(vocab)
        print('Vocab build complete!')
        with open('saved/word2id.pkl', 'wb') as f:
            pickle.dump(self.word2id, f)
        with open('saved/id2word.pkl', 'wb') as f:
            pickle.dump(self.id2word, f)
        print('Save complete!')
        return


