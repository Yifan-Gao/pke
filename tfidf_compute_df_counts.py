import logging
import sys
from string import punctuation
import os
from pke import compute_document_frequency
from pke.readers import str2spacy
import spacy

"""Compute Document Frequency (DF) counts from a collection of documents.

N-grams up to 3-grams are extracted and converted to their n-stems forms.
Those containing a token that occurs in a stoplist are filtered out.
Output file is in compressed (gzip) tab-separated-values format (tsv.gz).
"""

# setting info in terminal
logging.basicConfig(level=logging.INFO)

data_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/tfidf'

for lang in ['de', 'es', 'fr', 'it']:
    # path to the collection of documents
    input_dir = os.path.join(data_path, 'docs_train', f'{lang}')
    
    # path to the df weights dictionary, saved as a gzipped csv file
    output_file = os.path.join(data_path, f'{lang}.df_counts.csv.gz')
    
    # stoplist are punctuation marks
    stoplist = list(punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

    spacy_model = spacy.load(str2spacy(lang), disable=['ner', 'textcat', 'parser'])
    if int(spacy.__version__.split('.')[0]) < 3:
        sentencizer = spacy_model.create_pipe('sentencizer')
    else:
        sentencizer = 'sentencizer'
    spacy_model.add_pipe(sentencizer)

    # compute idf weights
    compute_document_frequency(input_dir=input_dir,
                               output_file=output_file,
                               extension='txt', # input file extension
                               language=lang, # language of the input files
                               normalization="stemming", # use porter stemmer
                               stoplist=stoplist,  # stoplist
                               delimiter='\t',  # tab separated output
                               n=3,  # compute n-grams up to 5-grams
                               spacy_model=spacy_model)