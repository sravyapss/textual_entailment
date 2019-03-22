import numpy
import theano
import theano.tensor as T

from lasagne import nonlinearities, init
from lasagne.layers.base import Layer, MergeLayer

import pdb

class Attention1(Layer):
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),b=init.Constant(0.),**kwargs):
        super(Attention1, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearities.tanh
        self.num_units = num_units
        num_inputs = self.input_shape[2]
	print(str(num_inputs) + '  1')
        self.W = self.add_param(W, (num_inputs, num_units))
        self.b = self.add_param(b, (num_units,), regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, **kwargs):
        ans = T.dot(input, self.W) + self.b.dimshuffle('x', 'x', 0)
        final = self.nonlinearity(ans)
        return final

class Attention2(Layer):
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),b=init.Constant(0.),**kwargs):
        super(Attention2, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearities.identity
        self.num_units = num_units
        num_inputs = self.input_shape[2]
	print(str(num_inputs) + ' 2')
        self.W = self.add_param(W, (num_inputs, num_units))
        self.b = self.add_param(b, (num_units,), regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, **kwargs):
        ans = T.dot(input, self.W) + self.b.dimshuffle('x', 'x', 0)
        final = self.nonlinearity(ans)
        return final

class Softmax(MergeLayer):
    def __init__(self, incoming, mask=None, **kwargs):
        incomings = []
        incomings.append(incoming)
        incomings.append(mask)
        super(Softmax, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2])

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        x = input.shape[0]
        y = input.shape[1]
        z = input.shape[2]
        mask = inputs[1]
        input = input * mask.dimshuffle(0, 1, 'x').astype(theano.config.floatX) - numpy.asarray(1e36).astype(theano.config.floatX) * (1 - mask).dimshuffle(0, 1, 'x').astype(theano.config.floatX)
        annotation = T.nnet.softmax(input.dimshuffle(0, 2, 1).reshape((x*z,y)))
        annotation = annotation.reshape((x,z,y)).dimshuffle(0,2,1)
        return annotation

class DotProduct(MergeLayer):
    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][2], input_shapes[1][2])

    def get_output_for(self, inputs, **kwargs):
        A, H = inputs[0], inputs[1]
        M = T.batched_dot(H.dimshuffle(0, 2, 1), A).dimshuffle(0, 2, 1)
        return M

class GatedEncoder(MergeLayer):
    def __init__(self, incomings, hidden, W1=init.GlorotUniform(), W2=init.GlorotUniform(), **kwargs):
        super(GatedEncoder, self).__init__(incomings, **kwargs)
        self.hidden_sent1 = self.input_shapes[0][2]
        self.hidden_sent2 = self.input_shapes[1][2]
        self.num_rows = self.input_shapes[0][1]
        self.hidden = hidden
        self.W1 = self.add_param(W1, (self.num_rows, self.hidden_sent1, self.hidden))
        self.W2 = self.add_param(W2, (self.num_rows, self.hidden_sent2, self.hidden))

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], self.hidden)

    def get_output_for(self, inputs, **kwargs):
        sent1, sent2 = inputs[0], inputs[1]
        sent1_out = T.batched_dot(T.extra_ops.cpu_contiguous(sent1.dimshuffle(1, 0, 2)), self.W1).dimshuffle(1, 0, 2)
        sent2_out = T.batched_dot(T.extra_ops.cpu_contiguous(sent2.dimshuffle(1, 0, 2)), self.W2).dimshuffle(1, 0, 2)
        gated_out = sent1_out * sent2_out
        return gated_out
