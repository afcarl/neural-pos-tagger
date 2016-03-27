import numpy as np

import io_utils


class Vocab(object):

    def __init__(self):
        self.i2w = []
        self.w2i = {}

    def add_word(self, word):
        assert isinstance(word, unicode)
        if word not in self.w2i:
            new_id = self.size()
            self.i2w.append(word)
            self.w2i[word] = new_id

    def get_id(self, word):
        assert isinstance(word, unicode)
        return self.w2i.get(word)

    def get_word(self, w_id):
        return self.i2w[w_id]

    def size(self):
        return len(self.i2w)

    def save(self, path):
        with open(path, 'w') as f:
            for i, w in enumerate(self.i2w):
                print >> f, str(i) + '\t' + w.encode('utf-8')

    @classmethod
    def load(cls, path):
        vocab = Vocab()
        with open(path) as f:
            for line in f:
                w = line.strip().split('\t')[1].decode('utf-8')
                vocab.add_word(w)
        return vocab


def convert_into_ids(corpus, vocab_word, vocab_char, vocab_tag):
    id_corpus_w = []
    id_corpus_c = []
    id_corpus_b = []
    id_corpus_t = []

    for sent in corpus:
        w_ids = []
        c_ids = []
        bs = []
        t_ids = []
        b = 0
        for w, t in sent:
            w_id = vocab_word.get_id(w.lower())
            t_id = vocab_tag.get_id(t)

            if w_id is None:
                w_id = vocab_word.get_id(io_utils.UNK)

            assert w_id is not None
            assert t_id is not None

            w_ids.append(w_id)
            t_ids.append(t_id)
            c_ids.extend([vocab_char.get_id(c) for c in w])
            b += len(w)
            bs.append(b)

        id_corpus_w.append(np.asarray(w_ids, dtype='int32'))
        id_corpus_c.append(np.asarray(c_ids, dtype='int32'))
        id_corpus_b.append(np.asarray(bs, dtype='int32'))
        id_corpus_t.append(np.asarray(t_ids, dtype='int32'))

    assert len(id_corpus_w) == len(id_corpus_c) == len(id_corpus_b) == len(id_corpus_t)
    return id_corpus_w, id_corpus_c, id_corpus_b, id_corpus_t
