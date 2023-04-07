#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_lemmas_from_synsets(lemma, pos) -> List:
    lemmas = []
    for synset in wn.synsets(lemma, pos):
        lemmas.extend(synset.lemmas())
    return lemmas

def get_candidates(lemma, pos) -> List[str]:
    # lemmas from wordnet
    lemmas = {lemma.name() for lemma in get_lemmas_from_synsets(lemma, pos)}

    # remove input lemma
    lemmas.remove(lemma)

    # correct compound words
    lemmas = {lemma.replace('_', ' ') for lemma in lemmas}

    return list(lemmas)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # lemmas from wordnet
    lemmas = get_lemmas_from_synsets(context.lemma, context.pos)

    # count total occurences of each word across all senses
    counts = {}
    for lemma in lemmas:
        counts[lemma.name()] = counts.setdefault(lemma.name(), 0) + lemma.count()

    # return highest count
    counts.pop(context.lemma)
    max_key, _ = max(counts.items(), key=lambda item: item[1])

    return max_key

def wn_simple_lesk_predictor(context : Context) -> str:
    return None #TODO part 3       
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        return None # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        return None # replace for part 5

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = smurf_predictor(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
