#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pke
import json
from pke.readers import str2spacy
import spacy
from tqdm import tqdm
import glob

for lang in ['de', 'es', 'fr', 'it']:
# for lang in ['de', ]:
    data_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline'

    input_dir = os.path.join(data_path, 'tfidf', 'docs_test', f'{lang}',)
    extension = 'txt'

    kea_model_file = os.path.join(data_path, "kea", f"model.{lang}.pickle")
    df_file = os.path.join(data_path, "tfidf", f'{lang}.df_counts.csv.gz')

    saving_path = os.path.join(data_path, 'kea')
    os.makedirs(saving_path, exist_ok=True)

    spacy_model = spacy.load(str2spacy(lang), disable=['ner', 'textcat', 'parser'])
    if int(spacy.__version__.split('.')[0]) < 3:
        sentencizer = spacy_model.create_pipe('sentencizer')
    else:
        sentencizer = 'sentencizer'
    spacy_model.add_pipe(sentencizer)

    predictions = {}

    for input_file in tqdm(glob.iglob(input_dir + os.sep + '*.' + extension)):

        # create a Kea extractor and set the input language to English (used for
        # the stoplist in the candidate selection method)
        extractor = pke.supervised.Kea()

        # load the content of the document, here in CoreNLP XML format
        # the use_lemmas parameter allows to choose using CoreNLP lemmas or stems
        # computed using nltk
        # extractor.load_document('C-1.xml')
        extractor.load_document(input=input_file,
                                language=lang,
                                normalization=None,
                                spacy_model=spacy_model)
        # select the keyphrase candidates, for Kea the 1-3 grams that do not start or
        # end with a stopword.
        extractor.candidate_selection()

        # load the df counts
        df_counts = pke.load_document_frequency_file(input_file=df_file,
                                                     delimiter='\t')

        # weight the candidates using Kea model.
        extractor.candidate_weighting(model_file=kea_model_file, df=df_counts)

        # print the n-highest (10) scored candidates
        curr_keyphrases = []
        for (keyphrase, score) in extractor.get_n_best(n=200, stemming=True):
            curr_keyphrases.append(keyphrase)
        predictions[input_file.split('.')[-2]] = curr_keyphrases

    with open(os.path.join(saving_path, f'predictions.{lang}.json'), 'w') as f:
        json.dump(predictions, f)
