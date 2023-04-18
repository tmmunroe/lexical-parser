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
    return lemma.lower().replace('_', ' ')

def get_lemmas_from_synsets(lemma, pos) -> List:
    lemmas = []
    for synset in wn.synsets(lemma, pos):
        for synset_lemma in synset.lemmas():
            if synset_lemma.name != lemma: 
                # only include lemmas that don't match the input
                lemmas.append(synset_lemma)
    return lemmas

def get_candidates(lemma:str, pos:str) -> List[str]:
    # lemmas from wordnet
    lemmas = {lemma.name() for lemma in get_lemmas_from_synsets(lemma, pos)}

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

def wn_simple_lesk_predictor(context : Context) -> str:
    # get lemmas for context word
    clean_name = clean_lemma_name(context.lemma)
    synsets = wn.synsets(context.lemma, context.pos)
    
    # calculate similarity scores for each lemma's synset
    context_str = ' '.join(context.left_context + [context.word_form] + context.right_context)
    lemma_synset_scores = []
    for synset in synsets:
        # get overlap
        context_set = set(sentence2tokens(context_str))
        joined_samples = set(sentence2tokens(' '.join(collect_synset_samples(synset))))
        synset_overlap = len(context_set & joined_samples)

        # get lemma frequency
        lemma_frequency = 0
        for lemma in synset.lemmas():
            if clean_lemma_name(lemma.name()) == clean_name:
                lemma_frequency = lemma.count()
                break
        else:
            raise ValueError('Expected to find lemma ', context.lemma, ' in synset ', synset)

        # get synset base score
        base_score = (10000*synset_overlap) + (100*lemma_frequency)

        # score lemmas in synset
        for lemma in synset.lemmas():
            if clean_lemma_name(lemma.name()) == clean_name: # skip context lemma since we only want replacements
                continue

            score = base_score + lemma.count()
            lemma_synset_scores.append((lemma, score))

    # figure out the lemma with the best synset, resolving ties
    best_lemma_scores = sorted(lemma_synset_scores, key=lambda item: item[-1], reverse=True)
    best_lemma, _ = best_lemma_scores[0]

    return clean_lemma_name(best_lemma.name())
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model:gensim.models.keyedvectors.KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        candidates = [c for c in get_candidates(context.lemma, context.pos) if self.model.has_index_for(c)]
        return self.model.most_similar_to_given(context.lemma, candidates)


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = set(get_candidates(context.lemma, context.pos))
        lemma_position = len(context.left_context)
        context_str = ' '.join(context.left_context + '[MASK]' + context.right_context)

        input_toks = self.tokenizer.encode(context_str)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][lemma_position])[::-1] # Sort in increasing order
        
        for best_word in best_words:
            token = self.tokenizer.convert_ids_to_tokens([best_word])[0]
            if token in candidates:
                return token

        return None

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).
    W2VMODEL_FILENAME = 'data/GoogleNews-vectors-negative300.bin.gz'

    # predictor = smurf_predictor
    predictor = wn_simple_lesk_predictor
    # predictor = wn_frequency_predictor
    # predictor = Word2VecSubst(W2VMODEL_FILENAME).predict_nearest
    # predictor = BertPredictor().predict

    
    for context in read_lexsub_xml(sys.argv[1]):
        print(context)  # useful for debugging
        prediction = predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
