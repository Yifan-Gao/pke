import string
import pke
import os
from tqdm import tqdm
import glob
import json
from pke.readers import str2spacy
import spacy

for lang in ['de', 'es', 'fr', 'it']:
# for lang in ['de', ]:
    data_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline'

    input_dir = os.path.join(data_path, 'tfidf', 'docs_test', f'{lang}',)
    extension = 'txt'

    df_file = os.path.join(data_path, "tfidf", f'{lang}.df_counts.csv.gz')

    saving_path = os.path.join(data_path, 'kpminer')
    os.makedirs(saving_path, exist_ok=True)

    spacy_model = spacy.load(str2spacy(lang), disable=['ner', 'textcat', 'parser'])
    if int(spacy.__version__.split('.')[0]) < 3:
        sentencizer = spacy_model.create_pipe('sentencizer')
    else:
        sentencizer = 'sentencizer'
    spacy_model.add_pipe(sentencizer)

    predictions = {}

    for input_file in tqdm(glob.iglob(input_dir + os.sep + '*.' + extension)):
        # 1. create a KPMiner extractor.
        extractor = pke.unsupervised.KPMiner()

        # 2. load the content of the document.
        extractor.load_document(input=input_file,
                                language=lang,
                                normalization=None,
                                spacy_model=spacy_model)

        # 3. select {1-5}-grams that do not contain punctuation marks or
        #    stopwords as keyphrase candidates. Set the least allowable seen
        #    frequency to 5 and the number of words after which candidates are
        #    filtered out to 200.
        lasf = 5
        cutoff = 200
        extractor.candidate_selection(lasf=lasf, cutoff=cutoff)

        # 4. weight the candidates using a `tf` x `idf`
        df = pke.load_document_frequency_file(input_file=df_file)
        alpha = 2.3
        sigma = 3.0
        extractor.candidate_weighting(df=df, alpha=alpha, sigma=sigma)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=200)
        predictions[input_file.split('.')[-2]] = [k[0] for k in keyphrases]

    with open(os.path.join(saving_path, f'predictions.{lang}.json'), 'w') as f:
        json.dump(predictions, f)

