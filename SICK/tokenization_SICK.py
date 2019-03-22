import sys
import string
import numpy
import cPickle
import numpy as np
import nltk
import pdb

embed = {}
idx_dict = {}
unseen = {}
classes = {'ENTAILMENT':0, 'NEUTRAL':1, 'CONTRADICTION':2, '-':3}
mapping = {0:'TRAIN', 1:'TRIAL', 2:'TEST'}
idx_dict[0] = 0
total_words = 1

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

def read_files(filename, embed, idx_dict, unseen, total_words, classes, mapping, flag):
    with open(filename, 'r') as file:
        sents = []
        head = file.readline()
        lines = file.readlines()
        cnt = 0
        for line in lines:
            line = line.split('\t')
            sent1 = nltk.word_tokenize(line[1])
            sent2 = nltk.word_tokenize(line[2])
	    #print(mapping[flag])
	    #print(line[11])
            if(line[11].strip().split('\n')[0] != mapping[int(flag)]):
                continue
            words1, idx_dict, total_words, unseen = get_word_embeddings(sent1, idx_dict, total_words, unseen, embed)
            words2, idx_dict, total_words, unseen = get_word_embeddings(sent2, idx_dict, total_words, unseen, embed)
            sents.append((numpy.asarray(words1).astype('int32'),
                          numpy.asarray(words2).astype('int32'),
                          numpy.asarray(classes[line[3]]).astype('int32')))
    print(cnt)
    return sents, idx_dict, total_words, unseen

with open('../glove.42B.300d.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        embedding = [float(x) for x in line[1:]]
        embed[line[0]] = np.asarray(embedding).astype('float32')

train = './SICK.txt'
val = './SICK.txt'
test = './SICK.txt'

all_data = []

sents, idx_dict, total_words, unseen = read_files(train, embed, idx_dict, unseen, total_words, classes, mapping, 0)
print(len(sents))
all_data.append(sents)
sents, idx_dict, total_words, unseen = read_files(val, embed, idx_dict, unseen, total_words, classes, mapping, 1)
print(len(sents))
all_data.append(sents)
sents, idx_dict, total_words, unseen = read_files(test, embed, idx_dict, unseen, total_words, classes, mapping, 2)
print(len(sents))
all_data.append(sents)

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
ordered_word_embedding = [numpy.zeros((1, 300), dtype='float32'), ] + \
    [unseen[inv_idx_dict[n]].reshape(1, -1) for n in range(1, len(inv_idx_dict))]
embeddings = numpy.concatenate(ordered_word_embedding, axis=0)

with open('./SICK_GloVe_converted', 'wb') as file:
    cPickle.dump(all_data, file)
    cPickle.dump(embeddings, file)
