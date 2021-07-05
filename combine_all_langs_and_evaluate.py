import json
import numpy as np
import string

def _normalize_keyphrase(kp):
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(kp)))


def _compute(predictions, references):
    """Returns the scores"""
    # TODO: Compute the different scores of the metric

    macro_metrics = {'precision': [], 'recall': [], 'f1': [], 'num_pred': [], 'num_gold': []}

    # for context, targets, preds in zip(context_lines, target_lines, preds_lines):
    for targets, preds in zip(references, predictions):
        targets = [_normalize_keyphrase(tmp_key).strip() for tmp_key in targets if
                   len(_normalize_keyphrase(tmp_key).strip()) != 0]
        preds = [_normalize_keyphrase(tmp_key).strip() for tmp_key in preds if
                 len(_normalize_keyphrase(tmp_key).strip()) != 0]

        total_tgt_set = set(targets)
        total_preds = set(preds)
        if len(total_tgt_set) == 0: continue

        # get the total_correctly_matched indicators
        total_correctly_matched = len(total_preds & total_tgt_set)

        # macro metric calculating
        precision = total_correctly_matched / len(total_preds) if len(total_preds) else 0.0
        recall = total_correctly_matched / len(total_tgt_set)
        f1 = 2 * precision * recall / (precision + recall) if total_correctly_matched > 0 else 0.0
        macro_metrics['precision'].append(precision * 100.0)
        macro_metrics['recall'].append(recall * 100.0)
        macro_metrics['f1'].append(f1 * 100.0)
        macro_metrics['num_pred'].append(len(total_preds))
        macro_metrics['num_gold'].append(len(total_tgt_set))

    return {
        "precision": np.mean(macro_metrics["precision"]),
        "recall": np.mean(macro_metrics["recall"]),
        "f1": np.mean(macro_metrics["f1"]),
        "num_pred": np.mean(macro_metrics["num_pred"]),
        "num_gold": np.mean(macro_metrics["num_gold"]),
        # "raw": macro_metrics,
    }

# pred_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/tfidf/predictions.{}.json'
# pred_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/kpminer/predictions.{}.json'
# pred_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/yake/predictions.{}.json'

# pred_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/textrank/predictions.{}.json'
# pred_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/topicrank/predictions.{}.json'
# pred_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/topicalpagerank/predictions.{}.json'
# pred_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/singlerank/predictions.{}.json'
# pred_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/positionrank/predictions.{}.json'
# pred_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/multipartiterank/predictions.{}.json'

# pred_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/kea/predictions.{}.json'

gold_path = '/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/{}/mix.test.json'

for pred_path in ['/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/tfidf/predictions.{}.json',
'/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/kpminer/predictions.{}.json',
'/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/yake/predictions.{}.json',
'/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/textrank/predictions.{}.json',
'/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/topicrank/predictions.{}.json',
'/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/topicalpagerank/predictions.{}.json',
'/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/singlerank/predictions.{}.json',
'/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/positionrank/predictions.{}.json',
'/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/multipartiterank/predictions.{}.json',
'/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2/percent-2/baseline/kea/predictions.{}.json',
]:


    all_preds = {}
    # for topk in [3,5,8,10,15,20,30,50,100,200]:
    for topk in [3,]:
        print(topk)
        predictions, references = [], []
        for lang in ['de', 'it', 'fr', 'es']:
            with open(pred_path.format(lang)) as f:
                pred = json.load(f)
            with open(gold_path.format(lang)) as f:
                gold = [json.loads(line) for line in f]
            asin2pred, asin2gold = {}, {}
            for idx, ex in enumerate(gold):
                asin2gold[ex['asin']] = ex['keywords'].split(';')
                asin2pred[ex['asin']] = pred[str(idx)][:topk]
                all_preds[ex['asin']] = pred[str(idx)]
            for asin in asin2gold:
                predictions.append(asin2pred[asin])
                references.append(asin2gold[asin])
        print(_compute(predictions, references))

    with open(pred_path.format('all'), 'w') as f:
        json.dump(all_preds, f)

