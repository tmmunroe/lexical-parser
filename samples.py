
# from NLTK
import nltk
from nltk.corpus import wordnet as wn

wn.lemmas('break', pos='n') # Retrieve all lexemes for the noun 'break'
l1 = wn.lemmas('break', pos='n')[0]
s1 = l1.synset() # get the synset for the first lexeme
s1
s1.lemmas() # Get all lexemes in that synset
s1.lemmas()[0].name() # Get the word of the first lexeme
s1.definition()
s1.examples()
s1.hypernyms()
s1.hyponyms()
l1.count() # Occurence frequency of this sense of 'break' in the SemCor corpus.


# from gensim
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True) 

# v1 = model.wv['computer']
v1 = model.get_vector('computer')
len(v1)

model.similarity('computer','calculator')


# numpy
import numpy as np

#cosine similarity
def cos_similarity(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

cos_similarity(model.get_vector('computer'),model.get_vector('calculator'))




