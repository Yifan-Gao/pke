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

    saving_path = os.path.join(data_path, 'textrank')
    os.makedirs(saving_path, exist_ok=True)

    spacy_model = spacy.load(str2spacy(lang), disable=['ner', 'textcat', 'parser'])
    if int(spacy.__version__.split('.')[0]) < 3:
        sentencizer = spacy_model.create_pipe('sentencizer')
    else:
        sentencizer = 'sentencizer'
    spacy_model.add_pipe(sentencizer)

    # define the set of valid Part-of-Speeches
    pos = {'NOUN', 'PROPN', 'ADJ'}

    predictions = {}

    for input_file in tqdm(glob.iglob(input_dir + os.sep + '*.' + extension)):
        # 1. create a TextRank extractor.
        extractor = pke.unsupervised.TextRank()

        # 2. load the content of the document.
        extractor.load_document(input=input_file,
                                language=lang,
                                normalization=None,
                                spacy_model=spacy_model)

        # 3. build the graph representation of the document and rank the words.
        #    Keyphrase candidates are composed from the 33-percent
        #    highest-ranked words.
        extractor.candidate_weighting(window=2,
                                      pos=pos,
                                      top_percent=0.33,
                                      )

        # 4. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=200)
        predictions[input_file.split('.')[-2]] = [k[0] for k in keyphrases]

    with open(os.path.join(saving_path, f'predictions.{lang}.json'), 'w') as f:
        json.dump(predictions, f)
