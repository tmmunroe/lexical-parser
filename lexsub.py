import sys

from lexsub_main import (
    smurf_predictor,
    wn_frequency_predictor,
    wn_simple_lesk_predictor,
    Word2VecSubst,
    BertPredictor
)

from lexsub_xml import read_lexsub_xml


def word2vec_predictor():
    W2VMODEL_FILENAME = 'data/GoogleNews-vectors-negative300.bin.gz'
    return Word2VecSubst(W2VMODEL_FILENAME).predict_nearest

def bert_predictor():
    return BertPredictor().predict

def resolve_predictor(predictor_name):
    predictor_map = {
        'smurf': smurf_predictor,
        'freq': wn_frequency_predictor,
        'lesk': wn_simple_lesk_predictor,
        'word2vec': word2vec_predictor,
        'bert': bert_predictor
    }

    if predictor_name not in predictor_map:
        raise ValueError(f'Could not resolve predictor {predictor_name}')
    return predictor_map[predictor_name]


if __name__=="__main__":
    predictor = resolve_predictor(sys.argv[1])
    for context in read_lexsub_xml('lexsub_trial.xml'):
        print(context)  # useful for debugging
        prediction = predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
