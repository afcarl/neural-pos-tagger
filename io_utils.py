import re
import gzip
import cPickle
from collections import defaultdict

import numpy as np

from preprocessor import Vocab

EOS = u'<EOS>'
UNK = u'<UNK>'

RE_NUM = re.compile(ur'[0-9]')


def load_init_emb(init_emb, init_emb_words, vocab):
    unk_id = vocab.get_id(UNK)

    # read first line and get dimension
    with open(init_emb) as f_emb:
        line = f_emb.readline()
        dim = len(line.split())
    assert dim > 0

    # initialize embeddings
    emb = np.random.randn(vocab.size(), dim).astype(np.float32)

    # corresponding IDs in given vocabulary
    ids = []

    with open(init_emb_words) as f_words:
        for i, line in enumerate(f_words):
            word = line.strip().decode('utf-8')

            # convert special characters
            if word == u'PADDING':
                word = EOS
            elif word == u'UNKNOWN':
                word = UNK

            w_id = vocab.get_id(word)

            # don't map unknown words to <UNK> unless it's really UNKNOWN
            if w_id == unk_id:
                if word == UNK:
                    ids.append(unk_id)
                else:
                    # no corresponding word in vocabulary
                    ids.append(None)
            else:
                ids.append(w_id)

    with open(init_emb) as f_emb:
        for i, emb_str in enumerate(f_emb):
            w_id = ids[i]
            if w_id is not None:
                emb[w_id] = emb_str.split()

    return emb


def load_conll(path, vocab_size=None, file_encoding='utf-8', limit_vocab=None):
    corpus = []
    word_freqs = defaultdict(int)
    char_freqs = defaultdict(int)
    max_char_len = -1

    vocab_word = Vocab()
    vocab_char = Vocab()
    vocab_tag = Vocab()
    vocab_word.add_word(EOS)
    vocab_word.add_word(UNK)
    vocab_char.add_word(EOS)

    with open(path) as f:
        wts = []
        for line in f:
            es = line.rstrip().split('\t')
            if len(es) == 10:
                word = es[1].decode(file_encoding)
                word = RE_NUM.sub(u'0', word)  # replace numbers with 0
                tag = es[4].decode(file_encoding)

                for c in word:
                    char_freqs[c] += 1

                max_char_len = len(word) if max_char_len < len(word) else max_char_len

                wt = (word, tag)
                wts.append(wt)
                word_freqs[word.lower()] += 1
                vocab_tag.add_word(tag)
            else:
                # reached end of sentence
                corpus.append(wts)
                wts = []
        if wts:
            corpus.append(wts)

    for w, f in sorted(word_freqs.items(), key=lambda (k, v): -v):
        if limit_vocab is None or w in limit_vocab:
            # register only words in limit_vocab
            if vocab_size is None or vocab_word.size() < vocab_size:
                vocab_word.add_word(w)
            else:
                break

    for c, f in sorted(char_freqs.items(), key=lambda (k, v): -v):
        vocab_char.add_word(c)

    return corpus, vocab_word, vocab_char, vocab_tag, max_char_len


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    if fn[-7:] != '.pkl.gz':
        fn += '.pkl.gz'

    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)
