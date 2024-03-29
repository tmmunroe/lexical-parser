from lexsub_main import (
    get_lemmas_from_synsets,
    get_candidates,
    wn_frequency_predictor,
    wn_simple_lesk_predictor,
    Word2VecSubst
)
from lexsub_xml import Context

def test_get_lemmas_from_synsets():
    lemmas = [lemma.name() for lemma in get_lemmas_from_synsets('slow','a')]
    expected = ['slow', 'slow', 'dense', 'dim', 'dull', 'dumb', 'obtuse',
                'slow', 'slow', 'boring', 'deadening', 'dull', 'ho-hum',
                'irksome', 'slow', 'tedious', 'tiresome', 'wearisome', 'dull',
                'slow', 'sluggish']
    assert lemmas.sort() == expected.sort()
    

def test_get_candidates():
    candidates = get_candidates('slow','a')
    expected = ['deadening', 'tiresome', 'sluggish', 'dense',
                'tedious', 'irksome', 'boring', 'wearisome',
                'obtuse', 'dim', 'dumb', 'dull', 'ho-hum']
    assert candidates.sort() == expected.sort()


def test_wn_frequency_predictor():
    context = Context(None, 'slow', 'slow', 'a', None, None)
    max_key = wn_frequency_predictor(context)
    expected = 'dumb'
    assert max_key == expected

def test_wn_simple_lesk_predictor():
    context = Context(
        None, 'examination', 
        'examination', 'n',
        ['all', 'questions'],
        ['were', 'difficult']
    )

    predicted = wn_simple_lesk_predictor(context)
    expected = 'scrutiny'
    assert predicted == expected

def test_word2vec():
    context = Context(None, 'slow', 'slow', 'a', None, None)
    wordvec_model = Word2VecSubst('data/GoogleNews-vectors-negative300.bin.gz')
    predicted = wordvec_model.predict_nearest(context)
    expected = 'sluggish'
    assert predicted == expected

def test_bert():
    context = Context(
        None, 'study',
        'study', 'n', 
        'they performed a rigourous'.split(),
        'on the effects of alcohol consumption'.split()
        )
    wordvec_model = Word2VecSubst('data/GoogleNews-vectors-negative300.bin.gz')
    predicted = wordvec_model.predict_nearest(context)
    expected = 'survey'
    assert predicted == expected