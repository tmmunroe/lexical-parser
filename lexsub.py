import sys

from lexsub_main import (
    smurf_predictor,
    wn_frequency_predictor,
    wn_simple_lesk_predictor,
    Word2VecSubst,
    BertPredictor,
    ensemble_predictor
)

from lexsub_xml import read_lexsub_xml


def word2vec_predictor():
    W2VMODEL_FILENAME = 'data/GoogleNews-vectors-negative300.bin.gz'
    return Word2VecSubst(W2VMODEL_FILENAME).predict_nearest

def bert_predictor():
    return BertPredictor().predict

def resolve_predictor(predictor_name):
    predictor_map = {
        'smurf': (smurf_predictor, False),
        'freq': (wn_frequency_predictor, False),
        'lesk': (wn_simple_lesk_predictor, False),
        'word2vec': (word2vec_predictor, True),
        'bert': (bert_predictor, True),
        'ensemble': (ensemble_predictor, True)
    }

    if predictor_name not in predictor_map:
        raise ValueError(f'Could not resolve predictor {predictor_name}')
    
    predictor, instantiate = predictor_map[predictor_name]
    if instantiate:
        predictor = predictor()
    return predictor


if __name__=="__main__":
    predictor = resolve_predictor(sys.argv[1])
    for context in read_lexsub_xml('lexsub_trial.xml'):
        # print(context)  # useful for debugging
        prediction = predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
