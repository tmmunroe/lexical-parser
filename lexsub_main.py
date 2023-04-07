#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

stemmer = PorterStemmer()
stop_words = stopwords.words('english')

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def sentence2tokens(sentence: str) -> List[str]:
    return [stemmer.stem(w) for w in word_tokenize(sentence) if w not in stop_words]

def clean_lemma_name(lemma: str) -> str:
    return lemma.replace('_', ' ')

def get_lemmas_from_synsets(lemma, pos) -> List:
    lemmas = []
    for synset in wn.synsets(lemma, pos):
        lemmas.extend(synset.lemmas())
    return lemmas

def get_candidates(lemma:str, pos:str) -> List[str]:
    # lemmas from wordnet
    lemmas = {lemma.name() for lemma in get_lemmas_from_synsets(lemma, pos)}

    # remove input lemma
    lemmas.remove(lemma)

    # correct compound words
    lemmas = {clean_lemma_name(lemma) for lemma in lemmas}

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

    return clean_lemma_name(max_key)

def collect_synset_samples(synset, extended=True) -> List[str]:
    # if extended, will also return results from hypernyms
    samples = [synset.definition()] + synset.examples()
    if extended:
        for hypernym in synset.hypernyms():
            samples.extend(collect_synset_samples(hypernym, extended=False))
    return samples

def synset_overlaps(context_str:str, synset) -> float:
    context_set = set(sentence2tokens(context_str))
    joined_samples = set(sentence2tokens(' '.join(collect_synset_samples(synset))))
    intersection = context_set & joined_samples
    # print(context_set)
    # print(joined_samples)
    # print(intersection)
    # print(len(intersection))
    # print(synset)
    # print([(l, l.count()) for l in synset.lemmas()])
    # print('--------------------------------------------------')
    # print('--------------------------------------------------')

    return len(intersection)

def top_scores(score_list):
    sorted_score_list = sorted(score_list, key=lambda item: item[1], reverse=True)
    best_score = sorted_score_list[0][1]
    best_end = 1
    for best_end in range(1, len(sorted_score_list)):
        if sorted_score_list[best_end][1] < best_score:
            return sorted_score_list[:best_end]
    return sorted_score_list

def wn_simple_lesk_predictor(context : Context) -> str:
    # get lemmas for context word
    lemmas = wn.lemmas(context.lemma, context.pos)
    if not lemmas:
        raise ValueError('No lemmas found for context: ', context)
    
    # calculate similarity scores for each lemma's synset
    context_str = ' '.join(context.left_context + [context.word_form] + context.right_context)
    lemma_overlap_scores = [(lemma, synset_overlaps(context_str, lemma.synset())) for lemma in lemmas]

    # figure out the lemma with the best synset, resolving ties
    best_lemma_scores = top_scores(lemma_overlap_scores)
    if len(best_lemma_scores) == 1:
        best_lemma, _ = best_lemma_scores[0]
    else: # tie - use the synset where the context lemma is most frequent
        best_lemma, _ = max(best_lemma_scores, key=lambda lemma_score: lemma_score[0].count())
    
    # get the synset synonym with the highest frequency 
    best_synonym_lemma, best_count = None, -1
    for lemma in best_lemma.synset().lemmas():
        if lemma.count() > best_count and lemma.name() != context.lemma:
            best_synonym_lemma = lemma
            best_count = lemma.count()

    return clean_lemma_name(best_synonym_lemma.name())
   

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
