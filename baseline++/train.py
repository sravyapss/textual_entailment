import time
import os
import sys
import numpy
import cPickle
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.recurrent import Gate
from lasagne import init, nonlinearities

from util import Attention1, Attention2, Softmax, DotProduct, DotProduct2, GatedEncoder
import pdb

LSTM_HIDDEN = int(sys.argv[1])          # 150 Hidden unit numbers in LSTM
ATTENTION_HIDDEN = int(sys.argv[2])     # 350 Hidden unit numbers in attention MLP
MLP_HIDDEN = int(sys.argv[3])           # 3000 Hidden unit numbers in output MLP
N_ROWS = int(sys.argv[4])               # 10 Number of rows in matrix representation
LEARNING_RATE = float(sys.argv[5])      # 0.01
ATTENTION_PENALTY = float(sys.argv[6])  # 1.
DROPOUT = float(sys.argv[7])             # 0.5 Dropout in GAE
WE_DIM = int(sys.argv[8])               # 300 Dim of word embedding
bsz = int(sys.argv[9])           # 50 Minibatch size
GRADIENT_CLIP = int(sys.argv[10])           # 100 All gradients above this will be clipped
epochs = int(sys.argv[11])          # 12 Number of epochs to train the net
STD = float(sys.argv[12])               # 0.1 Standard deviation of weights in initialization
max1 = 85
max2 = 85
filename = __file__.split('.')[0] + \
           '_LSTMHIDDEN' + str(LSTM_HIDDEN) + \
           '_ATTENTIONHIDDEN' + str(ATTENTION_HIDDEN) + \
           '_OUTHIDDEN' + str(MLP_HIDDEN) + \
           '_NROWS' + str(N_ROWS) + \
           '_LEARNINGRATE' + str(LEARNING_RATE) + \
           '_ATTENTIONPENALTY' + str(ATTENTION_PENALTY) + \
           '_GAEREG' + str(DROPOUT) + \
           '_WEDIM' + str(WE_DIM) + \
           '_BATCHSIZE' + str(bsz) + \
           '_GRADCLIP' + str(GRADIENT_CLIP) + \
           '_NUMEPOCHS' + str(epochs) + \
           '_STD' + str(STD)

def process(batch_data):
    label = []
    # max1 = 0; max2 = 0
    bs = len(batch_data)
    for item in batch_data:
        # max1 = max(max1,len(item[0]))
        # max2 = max(max2,len(item[1]))
        label.append(item[2])
    dat1=numpy.zeros((bs, max1), dtype='int32')
    mask1=numpy.zeros((bs, max1), dtype='int32')
    dat2=numpy.zeros((bs, max2), dtype='int32')
    mask2=numpy.zeros((bs, max2), dtype='int32')
    for i in range(len(batch_data)):
        item1 = batch_data[i][0]
        item2 = batch_data[i][1]
        dat1[i,:len(item1)] = item1
        mask1[i,:len(item1)] = (1,) * len(item1)
        dat2[i,:len(item2)] = item2
        mask2[i,:len(item2)] = (1,) * len(item2)

    return dat1, mask1, dat2, mask2, label


def train(epoch_no, data_train, compute_train_costerror):
    cost = 0.
    error = 0.
    for batch, i in enumerate(range(0, len(data_train)//5, bsz)):
    	if(i+bsz>=len(data_train)//5):
	    break
        sent1, mask1, sent2, mask2, targets = process(data_train[i:i+bsz])
        cost_inter, error_inter = compute_train_costerror(sent1,mask1,sent2,mask2,targets)
        cost += cost_inter
        error += error_inter
        if batch % 100 == 0:
            print("Sample %d %.2fs, train cost %f, error %f"  % (batch * bsz,LEARNING_RATE,cost,error))
    return cost, error


def evaluate(data, compute_costerror):
    cost = 0.
    error = 0.
    for batch, i in enumerate(range(0, len(data), bsz)):
        sent1, mask1, sent2, mask2, targets = process(data[i:min(i+bsz,len(data))])
        cost_inter, error_inter = compute_costerror(sent1,mask1,sent2,mask2,targets)
        cost += cost_inter
        error += error_inter
    return cost, error


def BiLSTM(embedding, mask, STD, LSTM_HIDDEN, GRADIENT_CLIP, back):
    lstm = lasagne.layers.LSTMLayer(
        embedding, mask_input=mask, num_units=LSTM_HIDDEN,
        ingate=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD),
                    W_cell=init.Normal(STD)),
        forgetgate=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD),
                    W_cell=init.Normal(STD)),
        cell=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD),
                  W_cell=None, nonlinearity=nonlinearities.tanh),
        outgate=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD),
                    W_cell=init.Normal(STD)),
        nonlinearity=lasagne.nonlinearities.tanh,
        peepholes = False,
        grad_clipping=GRADIENT_CLIP, backwards=back)
    return lstm

def main():
    file = open('../SNLI_GloVe_converted', 'rb')
    data_train, data_val, data_test = cPickle.load(file)
    embeddings =  cPickle.load(file).astype(theano.config.floatX)
    file.close()
    print(len(data_train))
    ##################### Build LSTM #####################

    sent = T.TensorType('int32', [False, False])
    mask = T.TensorType('int32', [False, False])
    input_layer = lasagne.layers.InputLayer(shape=(bsz, max1), input_var=sent)
    mask_layer = lasagne.layers.InputLayer(shape=(bsz, max1), input_var=mask)
    embedding_layer = lasagne.layers.EmbeddingLayer(input_layer, input_size = embeddings.shape[0], output_size = embeddings.shape[1], W = embeddings)
    forward_lstm = BiLSTM(embedding_layer, mask_layer, STD, LSTM_HIDDEN, GRADIENT_CLIP, False)
    backward_lstm = BiLSTM(embedding_layer, mask_layer, STD, LSTM_HIDDEN, GRADIENT_CLIP, True)
    final_lstm = lasagne.layers.ConcatLayer([forward_lstm, backward_lstm], axis=2)

    #################### Attention mechanism ###############

    # weight1_layer = Attention1(final_lstm, num_units = ATTENTION_HIDDEN)
    # weight2_layer = Attention2(weight1_layer, num_units = N_ROWS)
    # annotation_layer = Softmax(weight2_layer, mask=mask_layer)
    # sentence_embed_layer = DotProduct([annotation_layer, final_lstm])

    ################ Sent1 and Sent2 embeddings ###############

    sent1 = T.TensorType('int32', [False, False])('sentence_vector')
    sent2 = T.TensorType('int32', [False, False])('hypothesis_vector')
    sent1_layer = lasagne.layers.InputLayer(shape=(bsz, max1), input_var=sent1)
    sent2_layer = lasagne.layers.InputLayer(shape=(bsz, max1), input_var=sent2)

    mask1 = T.TensorType('int32', [False, False])('sentence_mask')
    mask2 = T.TensorType('int32', [False, False])('hypothesis_mask')
    mask1_layer = lasagne.layers.InputLayer(shape=(bsz, max1), input_var=mask1)
    mask2_layer = lasagne.layers.InputLayer(shape=(bsz, max1), input_var=mask2)

    # sent1_embedding, sent1_annotation = lasagne.layers.get_output([sentence_embed_layer, annotation_layer], {input_layer: sent1_layer.input_var, mask_layer: mask1_layer.input_var})
    # sent2_embedding, sent2_annotation = lasagne.layers.get_output([sentence_embed_layer, annotation_layer], {input_layer: sent2_layer.input_var, mask_layer: mask2_layer.input_var})

    sent1_lstm = lasagne.layers.get_output(final_lstm, {input_layer:sent1_layer.input_var, mask_layer: mask1_layer.input_var})
    sent2_lstm = lasagne.layers.get_output(final_lstm, {input_layer:sent2_layer.input_var, mask_layer: mask2_layer.input_var})

    sent1_lstm_layer = lasagne.layers.InputLayer(shape=(bsz, max1, 2*LSTM_HIDDEN), input_var=sent1_lstm)
    sent2_lstm_layer = lasagne.layers.InputLayer(shape=(bsz, max1, 2*LSTM_HIDDEN), input_var=sent2_lstm)

    # attention_hidden = lasagne.layers.ConcatLayer([sent1_lstm_layer, sent2_lstm_layer], axis=1)
    attention_hidden1 = DotProduct2([sent2_lstm_layer, sent1_lstm_layer])
    attention_hidden2 = DotProduct2([sent1_lstm_layer, sent2_lstm_layer])

    weight11_layer = Attention1(attention_hidden1, num_units = ATTENTION_HIDDEN)
    weight12_layer = Attention2(weight11_layer, num_units = N_ROWS)
    annotation1_layer = Softmax(weight12_layer, mask=mask1_layer)
    sentence1_embed_layer = DotProduct([annotation1_layer, sent1_lstm_layer])

    weight12_layer = Attention1(attention_hidden2, num_units = ATTENTION_HIDDEN)
    weight22_layer = Attention2(weight12_layer, num_units = N_ROWS)
    annotation2_layer = Softmax(weight22_layer, mask=mask2_layer)
    sentence2_embed_layer = DotProduct([annotation2_layer, sent2_lstm_layer])

    sent1_embedding, sent1_annotation = lasagne.layers.get_output([sentence1_embed_layer, annotation1_layer], {sent1_lstm_layer: sent1_lstm_layer.input_var, sent2_lstm_layer: sent2_lstm_layer.input_var, mask1_layer: mask1_layer.input_var})
    sent2_embedding, sent2_annotation = lasagne.layers.get_output([sentence2_embed_layer, annotation2_layer], {sent1_lstm_layer: sent1_lstm_layer.input_var, sent2_lstm_layer: sent2_lstm_layer.input_var, mask2_layer: mask2_layer.input_var})


    ################## Gated Encoder ###########################

    sent1_embed_layer = lasagne.layers.InputLayer(shape=(bsz, N_ROWS, 2*LSTM_HIDDEN), input_var=sent1_embedding)
    sent2_embed_layer = lasagne.layers.InputLayer(shape=(bsz, N_ROWS, 2*LSTM_HIDDEN), input_var=sent2_embedding)
    gated_layer = GatedEncoder([sent1_embed_layer, sent2_embed_layer], hidden=2*LSTM_HIDDEN)




    ######################## MLP #################################

    gated_drop_layer = lasagne.layers.DropoutLayer(gated_layer, p=DROPOUT, rescale=True)
    hidden_layer = lasagne.layers.DenseLayer(gated_drop_layer, num_units=MLP_HIDDEN, nonlinearity=lasagne.nonlinearities.rectify)
    hidden_drop_layer = lasagne.layers.DropoutLayer(hidden_layer, p=DROPOUT, rescale=True)
    MLP_output_layer = lasagne.layers.DenseLayer(hidden_drop_layer, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)

    ####################################################################################################################################
    print("Layers Created")
    targets = T.ivector('targets')
    final_out_train = lasagne.layers.get_output(MLP_output_layer)
    final_out = lasagne.layers.get_output(MLP_output_layer, deterministic=True)

    penalty_sent1 = T.mean((T.batched_dot(sent1_annotation, sent1_annotation.dimshuffle(0,2,1))- T.eye(sent1_annotation.shape[1]).dimshuffle('x',0,1))**2, axis=(0,1,2))
    penalty_sent2 = T.mean((T.batched_dot(sent2_annotation, sent2_annotation.dimshuffle(0,2,1))- T.eye(sent2_annotation.shape[1]).dimshuffle('x',0,1))**2, axis=(0,1,2))
    penalty = penalty_sent1 + penalty_sent2

    cost_train = T.mean(T.nnet.categorical_crossentropy(final_out_train, targets) + ATTENTION_PENALTY * penalty)
    cost = T.mean(T.nnet.categorical_crossentropy(final_out, targets) + ATTENTION_PENALTY * penalty)

    params = lasagne.layers.get_all_params(MLP_output_layer)+ lasagne.layers.get_all_params(final_lstm)+  lasagne.layers.get_all_params(sentence1_embed_layer) + lasagne.layers.get_all_params(sentence2_embed_layer)
    file = 'params'+ os.sep + filename + '.pkl'
    if os.path.isfile(file):
        print("Resuming from file: " + file)
        param_values = cPickle.load(open(file, 'rb'))
        for p, v in zip(params, param_values):
            p.set_value(v)
    updates = lasagne.updates.adagrad(cost_train, params, LEARNING_RATE)

    prediction_train = T.argmax(final_out_train, axis=1)
    prediction = T.argmax(final_out, axis=1)
    error_rate_train = T.sum(T.neq(prediction_train, targets))
    error_rate = T.sum(T.neq(prediction, targets))

    compute_train_costerror = theano.function([sent1_layer.input_var,mask1_layer.input_var,sent2_layer.input_var,mask2_layer.input_var,targets],[cost_train, error_rate_train], updates=updates)
    compute_costerror = theano.function([sent1_layer.input_var,mask1_layer.input_var,sent2_layer.input_var,mask2_layer.input_var,targets],[cost, error_rate])

    print("Starting Training")
    for i in range(epochs):
        cost_train, error_train = train(i, data_train, compute_train_costerror)
        param_values = [p.get_value() for p in params]
        cPickle.dump(param_values, open('params' + os.sep +  filename + '.pkl', 'wb'))
        cost_val, error_val = evaluate(data_val, compute_costerror)
        cost_test, error_test = evaluate(data_test, compute_costerror)
        error_train = 5*error_train/len(data_train)
        error_val = error_val/len(data_val)
        error_test = (error_test/len(data_test))
        # print("epoch %d,          error train %f dev %f test %f" % (i, error_train))

        print("epoch %d,          error train %f dev %f test %f" % (i, error_train, error_val, error_test))
if __name__ == '__main__':
    main()
