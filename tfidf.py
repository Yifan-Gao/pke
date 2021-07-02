import string
import pke
import os
from tqdm import tqdm
import glob
import json

# lang='de'
for lang in ['es', 'fr', 'it']:
    data_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline'

    input_dir = os.path.join(data_path, 'tfidf', 'docs', f'{lang}',)
    extension = 'txt'

    df_file = os.path.join(data_path, "tfidf", f'{lang}.df_counts.csv.gz')

    # 1. create a TfIdf extractor.
    extractor = pke.unsupervised.TfIdf()

    predictions = {}

    for input_file in tqdm(glob.iglob(input_dir + os.sep + '*.' + extension)):
        # 2. load the content of the document.
        extractor.load_document(input=input_file,
                                language=lang,
                                normalization=None)

        # 3. select {1-3}-grams not containing punctuation marks as candidates.
        extractor.candidate_selection(n=3, stoplist=list(string.punctuation))

        # 4. weight the candidates using a `tf` x `idf`
        df = pke.load_document_frequency_file(input_file=df_file)
        extractor.candidate_weighting(df=df)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)
        predictions[input_file.split('.')[-2]] = [k[0] for k in keyphrases]

    with open(os.path.join(data_path, 'tfidf', f'predictions.{lang}.json'), 'w') as f:
        json.dump(predictions, f)