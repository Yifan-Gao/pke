import pke
import os
import glob
import json
from tqdm import tqdm
from pke.readers import str2spacy
import spacy

data_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline'

for lang in ['de', 'es', 'fr', 'it']:

    input_dir = os.path.join(data_path, 'tfidf', 'docs_test', f'{lang}',)
    extension = 'txt'

    saving_path = os.path.join(data_path, 'positionrank')
    os.makedirs(saving_path, exist_ok=True)

    spacy_model = spacy.load(str2spacy(lang), disable=['ner', 'textcat', 'parser'])
    if int(spacy.__version__.split('.')[0]) < 3:
        sentencizer = spacy_model.create_pipe('sentencizer')
    else:
        sentencizer = 'sentencizer'
    spacy_model.add_pipe(sentencizer)

    # define the valid Part-of-Speeches to occur in the graph
    pos = {'NOUN', 'PROPN', 'ADJ'}

    # define the grammar for selecting the keyphrase candidates
    grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

    predictions = {}

    for input_file in tqdm(glob.iglob(input_dir + os.sep + '*.' + extension)):
        # 1. create a PositionRank extractor.
        extractor = pke.unsupervised.PositionRank()

        # 2. load the content of the document.
        extractor.load_document(input=input_file,
                                language=lang,
                                normalization=None,
                                spacy_model=spacy_model)

        # 3. select the noun phrases up to 3 words as keyphrase candidates.
        extractor.candidate_selection(grammar=grammar,
                                      maximum_word_number=3)

        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk biaised with the position of the words
        #    in the document. In the graph, nodes are words (nouns and
        #    adjectives only) that are connected if they occur in a window of
        #    10 words.
        extractor.candidate_weighting(window=10,
                                      pos=pos)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=200)
        predictions[input_file.split('.')[-2]] = [k[0] for k in keyphrases]

    with open(os.path.join(saving_path, f'predictions.{lang}.json'), 'w') as f:
        json.dump(predictions, f)





