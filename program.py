# Source: src/doc2vec.py
# -*- coding: utf-8 -*-
from __future__ import division
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from random import shuffle
from sklearn.cross_validation import train_test_split
import nltk
import numpy as np

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
    
def tokenize_tags(label):
    tags = label.split("::")
    tags = map(lambda tok: mark_tag(tok), tags)
    return tags

def jaccard_similarity(labels, preds):
    lset = set(labels)
    pset = set(preds)
    return len(lset.intersection(pset)) / len(lset.union(pset))

def mark_tag(s):
    return "_" + s.replace(" ", "_")
    
def unmark_tag(s):
    return s[1:].replace("_", " ")
    
# read input data
orig_sents = []
sentences = []
filename = "/home/thej/Downloads/word2vec/labelled-data-final.txt"
with open(filename, "rt") as f:
    for line in f:
        qid,quest,label = line.split(',')
        orig_sents.append(quest)
        tokens = tokenize_text(quest)
        tags = tokenize_tags(label)
        sentences.append(LabeledSentence(words=tokens, tags=tags))
f.close()


train_sents, test_sents = train_test_split(sentences, test_size=0.1, 
                                           random_state=42)                

# PV-DBOW
model = Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2)

model.build_vocab(sentences)

alpha = 0.025
min_alpha = 0.025
num_epochs = 200
alpha_delta = (alpha - min_alpha) / num_epochs

for epoch in range(num_epochs):
    #shuffle(sentences)
    model.train(sentences)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay


tot_sim = 0.0
for test_sent in test_sents:
    pred_vec = model.infer_vector(test_sent.words)
    actual_tags = map(lambda x: unmark_tag(x), test_sent.tags)
    pred_tags = model.docvecs.most_similar([pred_vec], topn=5)
    pred_tags = filter(lambda x: x[0].find("_") > -1, pred_tags)
    pred_tags = map(lambda x: (unmark_tag(x[0]), x[1]), pred_tags)
    sim = jaccard_similarity(actual_tags, [x[0] for x in pred_tags])
    tot_sim += sim
print "Average Similarity on Test Set: %.3f" % (tot_sim / len(test_sents)) 

pred_vec_what = model.infer_vector("what time do we go to the mall ?")
pred_tags_what = model.docvecs.most_similar([pred_vec_what], topn=5)
print pred_tags_what

pred_vec_when = model.infer_vector("what time does the flight leave ? ?")
pred_tags_when = model.docvecs.most_similar([pred_vec_when], topn=5)
print pred_tags_when

