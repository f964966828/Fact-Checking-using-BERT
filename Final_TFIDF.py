from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np
import json
import glob

ARTICLE_FOLDER_PATH = './2022-inlp-final/articles/'
TFIDF_THRESHOLD = 0.2
UPPERCASE_RATIO_TOLERANT = 1.25


def load_json(path):
    with open(path, 'r', encoding='utf-8') as jsonfile:
        content = json.load(jsonfile)
    return content


def get_claim_tfidf(claim):
    claim_tfidf_model = TfidfVectorizer(stop_words='english')
    claim_tfidf_matrix = claim_tfidf_model.fit_transform(claim).toarray()
    vocab_list = claim_tfidf_model.get_feature_names_out()
    result = {}
    for vocab, score in zip(vocab_list, claim_tfidf_matrix[0]):
        result[vocab] = score
    return result


def get_article_importance(str_list, claim_tfidf):
    claim_vocab_list = list(claim_tfidf)
    claim_vocab_pos = dict()
    tfidf_model = TfidfVectorizer(stop_words='english')
    try:
        tfidf_fit = tfidf_model.fit_transform(str_list).toarray()
    except ValueError:
        return None
    vocab_dict = tfidf_model.vocabulary_
    for vocab in claim_vocab_list:
        if vocab_dict.get(vocab) is not None:
            claim_vocab_pos[vocab] = vocab_dict[vocab]

    result = [0] * tfidf_fit.shape[0]
    for i, arr in enumerate(tfidf_fit):
        for vocab, pos in claim_vocab_pos.items():
            result[i] += arr[pos] * claim_tfidf[vocab]
    return result


def get_evidence_vector(claim, data_id):
    claim_tfidf_dict = get_claim_tfidf(claim)
    json_list = glob.glob(ARTICLE_FOLDER_PATH + f'{data_id}_*')
    result = []
    for json_path in json_list:
        article = load_json(json_path)
        tfidf_list = get_article_importance(article, claim_tfidf_dict)

        if tfidf_list is None:
            continue

        idxs = []
        last_idx = len(tfidf_list) - 1
        for i, score in enumerate(tfidf_list):
            if score > TFIDF_THRESHOLD:
                idxs.append(i)
                if i > 0:
                    idxs.append(i-1)
                if i < last_idx:
                    idxs.append(i+1)
        idxs = np.unique(np.array(idxs))
        evidence = ' '.join([article[idx] for idx in idxs if len(article[idx].split()) > 2])
        
        if evidence != '':
            result.append(evidence)

    return result


def filter_evidence(evidences):
    result = []
    for evidence in evidences:
        word_cnt = len(evidence.split())
        ratio = sum(1 for c in evidence if c.isupper())/word_cnt
        if ratio < UPPERCASE_RATIO_TOLERANT:
            result.append(evidence)
    return result

def get_evidence_json(mode):

    data = load_json(f'./2022-inlp-final/{mode}.json')
    evidence_data = {}

    for i in tqdm(range(len(data))):
        task = data[i]
        evidence_vector = get_evidence_vector([task['metadata']['claim']], task['metadata']['id'])
        evidence_vector = filter_evidence(evidence_vector)
        evidence_data[task['metadata']['id']] = evidence_vector

    with open(f'./2022-inlp-final/{mode}_evidence.json', 'w', encoding='utf-8') as fout:
        json.dump(evidence_data, fout, ensure_ascii=False, indent=4)

get_evidence_json(mode='train')
get_evidence_json(mode='valid')
get_evidence_json(mode='test')
