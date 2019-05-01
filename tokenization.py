import sys
import string
import numpy
import cPickle
import numpy as np
import nltk
import pdb
from gensim.models import FastText
import os

embed = {}
idx_dict = {}
unseen = {}
classes = {'entailment':0, 'neutral':1, 'contradiction':2, '-':3}
#limit_train = int(sys.argv[1])
idx_dict[0] = 0
total_words = 1
longest_pre = 0
longest_hyp = 0


def get_word_embeddings(sent, idx_dict, total_words, unseen, embed):

    words = []
    for word in sent:
        word = word.strip(string.punctuation)
        if word not in idx_dict:
            idx_dict[word] = total_words
            idx = total_words
            total_words += 1
        else:
            idx = idx_dict[word]
        words.append(idx)
        if word not in embed:
            if word not in unseen:
                unseen[word] = []
            for others in words:
                if others in embed:
                    unseen[word].append(embed[others])
    return words, idx_dict, total_words, unseen

def read_files(filename, embed, idx_dict, unseen, total_words, classes):
    global longest_pre
    global longest_hyp
    with open(filename, 'r') as file:
        sents = []
        head = file.readline()
        lines = file.readlines()
        cnt = 0
        for line in lines:
            cnt +=1
            line = line.split('\t')
            sent1 = nltk.word_tokenize(line[5])
            if(len(sent1)>longest_hyp):
                longest_hyp = len(sent1)
            sent2 = nltk.word_tokenize(line[6])
            if(len(sent2)>longest_pre):
                longest_pre = len(sent2)
            if line[0]=='-':
                continue
            words1, idx_dict, total_words, unseen = get_word_embeddings(sent1, idx_dict, total_words, unseen, embed)
            words2, idx_dict, total_words, unseen = get_word_embeddings(sent2, idx_dict, total_words, unseen, embed)
            sents.append((numpy.asarray(words1).astype('int32'),
                          numpy.asarray(words2).astype('int32'),
                          numpy.asarray(classes[line[0]]).astype('int32')))
    return sents, idx_dict, total_words, unseen

def get_sents(filename):
    sentence_ted = []
    with open(filename, 'r') as file:

        head = file.readline()
        lines = file.readlines()
        cnt = 0
        for line in lines:
            cnt +=1
            line = line.split('\t')
            sent1 = nltk.word_tokenize(line[5])
            sent2 = nltk.word_tokenize(line[6])
            if line[0]=='-':
                continue

            senta1 = []
            for word in sent1:
                word = word.strip(string.punctuation)
                senta1.append(word)
            sentence_ted.append(senta1)

            senta2 = []
            for word in sent2:
                word = word.strip(string.punctuation)
                senta2.append(word)
            sentence_ted.append(senta2)
    return sentence_ted

def tokenize(model, invdict):
    embeds = []
    for n in range(1, len(invdict)):
        embeds.append(model.wv[invdict[n]].reshape(1, -1))

    return embeds



with open('../../glove.42B.300d.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        embedding = [float(x) for x in line[1:]]
        embed[line[0]] = np.asarray(embedding).astype('float32')

train = '../snli_1.0/snli_1.0/snli_1.0_train.txt'
val = '../snli_1.0/snli_1.0/snli_1.0_dev.txt'
test = '../snli_1.0/snli_1.0/snli_1.0_test.txt'

all_data = []

sents, idx_dict, total_words, unseen = read_files(train, embed, idx_dict, unseen, total_words, classes)
all_data.append(sents)
sents, idx_dict, total_words, unseen = read_files(val, embed, idx_dict, unseen, total_words, classes)
all_data.append(sents)
sents, idx_dict, total_words, unseen = read_files(test, embed, idx_dict, unseen, total_words, classes)
all_data.append(sents)


sentence_ted = get_sents(train)
# sentence_ted.append(sentstr)
sentsval = get_sents(val)
sentence_ted.extend(sentsval)
sentstest = get_sents(test)
sentence_ted.extend(sentstest)




mean_words = np.zeros((300,))
mean_words_std = 1e-1
npy_rng = np.random.RandomState(123)

for word in unseen:
    if len(unseen[word]) != 0:
        unseen[word] = sum(unseen[word]) / len(unseen[word])
    else:
        unseen[word] = mean_words + npy_rng.randn(mean_words.shape[0]) * \
                             mean_words_std * 0.1

unseen.update(embed)
inv_idx_dict = {v: k for k, v in idx_dict.items()}


exists = os.path.isfile('token_model')
if exists:
    model = FastText.load('token_model')
    emb = tokenize(model, inv_idx_dict)

    ordered_word_embedding = [numpy.zeros((1, 300), dtype='float32'), ] + emb
    embeddings = numpy.concatenate(ordered_word_embedding, axis=0)

    with open('../SNLI_GloVe_converted', 'wb') as file:
        cPickle.dump(all_data, file)
        cPickle.dump(embeddings, file)
else:
    model = FastText(sentence_ted, size=300, min_count=1, sg=1)
    model.save('token_model')


    emb = tokenize(model, inv_idx_dict)

    ordered_word_embedding = [numpy.zeros((1, 300), dtype='float32'), ] + emb
    embeddings = numpy.concatenate(ordered_word_embedding, axis=0)

    with open('../SNLI_GloVe_converted', 'wb') as file:
        cPickle.dump(all_data, file)
        cPickle.dump(embeddings, file)
print(longest_hyp)
print(longest_pre)
