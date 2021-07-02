# -*- coding: utf-8 -*-

import logging
import os
import pke

from pke.readers import str2spacy
import spacy

# setting info in terminal
logging.basicConfig(level=logging.INFO)

data_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline'

for lang in ['de', 'es', 'fr', 'it']:

    input_dir = os.path.join(data_path, 'tfidf', 'docs_train', f'{lang}',)
    extension = 'txt'

    saving_path = os.path.join(data_path, 'kea')
    os.makedirs(saving_path, exist_ok=True)

    spacy_model = spacy.load(str2spacy(lang), disable=['ner', 'textcat', 'parser'])
    if int(spacy.__version__.split('.')[0]) < 3:
        sentencizer = spacy_model.create_pipe('sentencizer')
    else:
        sentencizer = 'sentencizer'
    spacy_model.add_pipe(sentencizer)

    # path to the reference file
    reference_file = os.path.join(data_path, 'kea', f"gold_annotation_{lang}.txt")

    # path to the df file
    df_file = os.path.join(data_path, "tfidf", f'{lang}.df_counts.csv.gz')
    logging.info('Loading df counts from {}'.format(df_file))
    df_counts = pke.load_document_frequency_file(input_file=df_file,
                                                 delimiter='\t')

    # path to the model, saved as a pickle
    output_mdl = os.path.join(saving_path, f"model.{lang}.pickle")

    pke.train_supervised_model(input_dir=input_dir,
                               reference_file=reference_file,
                               model_file=output_mdl,
                               extension='txt',
                               language=lang,
                               normalization="stemming",
                               df=df_counts,
                               model=pke.supervised.Kea(),
                               spacy_model=spacy_model)
