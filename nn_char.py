__author__ = 'hiroki'

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.conv import conv2d

from nn_utils import sample_weights, relu
from optimizers import sgd, ada_grad


class Model(object):
    def __init__(self, x, c, b, y, opt, lr, init_emb, vocab_size, char_size, window, n_emb, n_c_emb, n_h, n_c_h, n_y, reg=0.0001):
        """
        :param n_emb: dimension of word embeddings
        :param window: window size
        :param n_h: dimension of hidden layer
        :param n_y: number of tags
        x: 1D: batch size * window, 2D: emb_dim
        h: 1D: batch_size, 2D: hidden_dim
        """

        assert window % 2 == 1, 'Window size must be odd'

        """ input """
        self.x = x
        self.c = c
        self.b = b
        self.y = y

        n_phi = n_emb + n_c_emb * window
        n_words = x.shape[0]

        """ params """
        if init_emb is not None:
            self.emb = theano.shared(init_emb)
        else:
            self.emb = theano.shared(sample_weights(vocab_size, n_emb))
        self.emb_c = theano.shared(sample_weights(char_size, n_c_emb))
        self.W_in = theano.shared(sample_weights(n_h, window * n_phi))
        self.W_in_c = theano.shared(sample_weights(window * n_c_emb, n_c_h))
        self.W_out = theano.shared(sample_weights(n_h, n_y))
        self.params = [self.emb_c, self.W_in, self.W_in_c, self.W_out]

        """ pad """
        self.zero = theano.shared(np.zeros(shape=(window / 2, n_phi), dtype=theano.config.floatX))
        self.zero_c = theano.shared(np.zeros(shape=(window / 2, n_c_emb), dtype=theano.config.floatX))

        """ look up embedding """
        self.x_emb = self.emb[self.x]  # x_emb: 1D: n_words, 2D: n_emb
        self.c_emb = self.emb_c[self.c]  # c_emb: 1D: n_char of a sent, 2D: n_c_emb

        """ create feature """
        self.c_phi = self.create_char_feature()
        self.x_phi = T.concatenate([self.x_emb, self.c_phi], axis=1)

        """ padding """
        self.x_padded = T.concatenate([self.zero, self.x_phi, self.zero], axis=0)  # x_padded: 1D: n_words + n_pad, 2D: n_phi

        """ convolution """
        self.x_u = self.x_padded.reshape((1, -1)).T
        self.x_in = conv2d(input=self.x_u, filters=self.W_in.dimshuffle(0, 1, 'x'), subsample=(n_phi, 1)).dimshuffle(1, 0, 2).reshape((n_words, -1))

        """ feed-forward computation """
        self.h = relu(self.x_in)
        self.p_y_given_x = T.nnet.softmax(T.dot(self.h, self.W_out))

        """ prediction """
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.neq(self.y_pred, self.y)

        """ cost function """
        self.nll = -T.sum(T.log(self.p_y_given_x)[T.arange(n_words), self.y])

#        self.L2_sqr = (self.emb ** 2).sum()
#        for p in self.params:
#            self.L2_sqr += (p ** 2).sum()

        self.cost = self.nll  # + reg * self.L2_sqr / 2

        if opt == 'sgd':
            self.updates = sgd(self.cost, self.params, self.emb, self.x_emb, lr)
        else:
            self.updates = ada_grad(self.cost, self.params, self.emb, self.x_emb, self.x, lr)

    def create_char_feature(self):
        def forward(b_t, b_tm1, c_emb, zero, W):
            c_tmp = c_emb[b_tm1: b_t]
            c_padded = T.concatenate([zero, c_tmp, zero], axis=0).reshape((1, -1)).T
            c_conv = conv2d(input=c_padded, filters=W)  # c_conv: 1D: n_c_h, 2D: n_char * slide
            c_t = T.max(c_conv.reshape((c_conv.shape[0], c_conv.shape[1])), axis=1)
            return c_t, b_t

        [c, _], _ = theano.scan(fn=forward,
                                sequences=[self.b],
                                outputs_info=[None, T.cast(0, 'int32')],
                                non_sequences=[self.c_emb, self.zero_c, self.W_in_c.dimshuffle(0, 1, 'x')])

        return c
