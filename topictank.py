#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this example uses TopicRank
import pke
import os
import glob
import json
from tqdm import tqdm
from pke.readers import str2spacy
import spacy
from pke.unsupervised import TopicRank


data_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline'

for lang in ['de', 'es', 'fr', 'it']:
    input_dir = os.path.join(data_path, 'tfidf', 'docs_test', f'{lang}', )
    extension = 'txt'

    saving_path = os.path.join(data_path, 'topicrank')
    os.makedirs(saving_path, exist_ok=True)

    spacy_model = spacy.load(str2spacy(lang), disable=['ner', 'textcat', 'parser'])
    if int(spacy.__version__.split('.')[0]) < 3:
        sentencizer = spacy_model.create_pipe('sentencizer')
    else:
        sentencizer = 'sentencizer'
    spacy_model.add_pipe(sentencizer)

    predictions = {}

    for input_file in tqdm(glob.iglob(input_dir + os.sep + '*.' + extension)):
        # create a TopicRank extractor
        extractor = TopicRank()

        # load the content of the document, here in CoreNLP XML format
        # the input language is set to English (used for the stoplist)
        # normalization is set to stemming (computed with Porter's stemming algorithm)
        extractor.load_document(input=input_file,
                                language=lang,
                                normalization='stemming',
                                spacy_model=spacy_model)

        # select the keyphrase candidates, for TopicRank the longest sequences of
        # nouns and adjectives
        extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})

        # weight the candidates using a random walk. The threshold parameter sets the
        # minimum similarity for clustering, and the method parameter defines the
        # linkage method
        extractor.candidate_weighting(threshold=0.74,
                                      method='average')

        # print the n-highest (10) scored candidates
        curr_keyphrases = []
        for (keyphrase, score) in extractor.get_n_best(n=200, stemming=True):
            curr_keyphrases.append(keyphrase)
        predictions[input_file.split('.')[-2]] = curr_keyphrases

    with open(os.path.join(saving_path, f'predictions.{lang}.json'), 'w') as f:
        json.dump(predictions, f)
