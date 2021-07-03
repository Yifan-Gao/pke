import string
import pke
import os
from tqdm import tqdm
import glob
import json
from pke.readers import str2spacy
import spacy
from nltk.corpus import stopwords

for lang in ['de', 'es', 'fr', 'it']:
# for lang in ['de', ]:
    data_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline'

    input_dir = os.path.join(data_path, 'tfidf', 'docs_test', f'{lang}',)
    extension = 'txt'

    df_file = os.path.join(data_path, "tfidf", f'{lang}.df_counts.csv.gz')

    saving_path = os.path.join(data_path, 'yake')
    os.makedirs(saving_path, exist_ok=True)

    spacy_model = spacy.load(str2spacy(lang), disable=['ner', 'textcat', 'parser'])
    if int(spacy.__version__.split('.')[0]) < 3:
        sentencizer = spacy_model.create_pipe('sentencizer')
    else:
        sentencizer = 'sentencizer'
    spacy_model.add_pipe(sentencizer)

    predictions = {}

    for input_file in tqdm(glob.iglob(input_dir + os.sep + '*.' + extension)):
        # 1. create a YAKE extractor.
        extractor = pke.unsupervised.YAKE()

        # 2. load the content of the document.
        extractor.load_document(input=input_file,
                                language=lang,
                                normalization=None,
                                spacy_model=spacy_model)

        # 3. select {1-3}-grams not containing punctuation marks and not
        #    beginning/ending with a stopword as candidates.
        stoplist = stopwords.words('english')
        extractor.candidate_selection(n=3, stoplist=stoplist)

        # 4. weight the candidates using YAKE weighting scheme, a window (in
        #    words) for computing left/right contexts can be specified.
        window = 2
        use_stems = False  # use stems instead of words for weighting
        extractor.candidate_weighting(window=window,
                                      stoplist=stoplist,
                                      use_stems=use_stems)

        # 5. get the 10-highest scored candidates as keyphrases.
        #    redundant keyphrases are removed from the output using levenshtein
        #    distance and a threshold.
        threshold = 0.8
        keyphrases = extractor.get_n_best(n=200, threshold=threshold)
        predictions[input_file.split('.')[-2]] = [k[0] for k in keyphrases]

    with open(os.path.join(saving_path, f'predictions.{lang}.json'), 'w') as f:
        json.dump(predictions, f)



