import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d

from nn_utils import sample_weights, relu
from optimizers import sgd, ada_grad


class Model(object):
    def __init__(self, name, x, y, lr, init_emb, vocab_size, emb_dim, hidden_dim, output_dim, window, opt):

        assert window % 2 == 1, 'Window size must be odd'

        """ input """
        self.name = name
        self.x = x
        self.y = y
        self.lr = lr
        self.input = [self.x, self.y, self.lr]

        n_words = x.shape[0]

        """ params """
        if init_emb is not None:
            self.emb = theano.shared(init_emb)
        else:
            self.emb = theano.shared(sample_weights(vocab_size, emb_dim))

        self.W_in = theano.shared(sample_weights(hidden_dim, 1, window, emb_dim))
        self.W_out = theano.shared(sample_weights(hidden_dim, output_dim))

        self.b_in = theano.shared(sample_weights(hidden_dim, 1))
        self.b_y = theano.shared(sample_weights(output_dim))

        self.params = [self.W_in, self.W_out, self.b_in, self.b_y]

        """ pad """
        self.zero = theano.shared(np.zeros(shape=(1, 1, window / 2, emb_dim), dtype=theano.config.floatX))

        """ look up embedding """
        self.x_emb = self.emb[self.x]  # x_emb: 1D: n_words, 2D: n_emb

        """ convolution """
        self.x_in = self.conv(self.x_emb)

        """ feed-forward computation """
        self.h = relu(self.x_in.reshape((self.x_in.shape[1], self.x_in.shape[2])) + T.repeat(self.b_in, T.cast(self.x_in.shape[2], 'int32'), 1)).T
        self.o = T.dot(self.h, self.W_out) + self.b_y
        self.p_y_given_x = T.nnet.softmax(self.o)

        """ prediction """
        self.y_pred = T.argmax(self.o, axis=1)
        self.result = T.eq(self.y_pred, self.y)

        """ cost function """
        self.nll = -T.sum(T.log(self.p_y_given_x)[T.arange(n_words), self.y])
        self.cost = self.nll

        if opt == 'sgd':
            self.updates = sgd(self.cost, self.params, self.emb, self.x_emb, self.lr)
        else:
            self.updates = ada_grad(self.cost, self.params, self.emb, self.x_emb, self.x, self.lr)

    def conv(self, x_emb):
        x_padded = T.concatenate([self.zero, x_emb.reshape((1, 1, x_emb.shape[0], x_emb.shape[1])), self.zero], axis=2)  # x_padded: 1D: n_words + n_pad, 2D: n_phi
        return conv2d(input=x_padded, filters=self.W_in)
