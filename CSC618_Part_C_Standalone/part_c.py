"""Real Cranfield experiment: TF-IDF, BM25, Dirichlet query-likelihood LM.

Run: pip install ir_datasets scikit-learn numpy; python part_c.py
"""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean

import ir_datasets
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluation import evaluate, ndcg_at_k

OUT = Path(__file__).parent / 'results'
OUT.mkdir(exist_ok=True)
K = 10
TOKEN = re.compile(r'(?u)\b\w\w+\b')


def tokens(s):
    return TOKEN.findall(s.lower())


def main():
    dataset = ir_datasets.load('cranfield')
    docs = list(dataset.docs_iter())
    queries = list(dataset.queries_iter())
    qrels = list(dataset.qrels_iter())
    ids = [d.doc_id for d in docs]
    texts = [d.title + ' ' + d.text for d in docs]
    bags = [Counter(tokens(t)) for t in texts]
    lengths = np.asarray([sum(c.values()) for c in bags], dtype=float)
    vocab_counts = sum(bags, Counter())
    collection_len = sum(vocab_counts.values())
    docfreq = Counter(t for b in bags for t in b)
    postings = defaultdict(list)
    for i, bag in enumerate(bags):
        for word, tf in bag.items(): postings[word].append((i,tf))
    avgdl = float(lengths.mean())

    # Consistent lowercase, token-length >= 2; no stemming or stopword deletion.
    tfidf = TfidfVectorizer(lowercase=True, token_pattern=TOKEN.pattern,
                            norm='l2', use_idf=True, sublinear_tf=False)
    matrix = tfidf.fit_transform(texts)

    grades = defaultdict(dict)
    for j in qrels:
        grades[j.query_id][j.doc_id] = max(j.relevance, 0)
    relevant = {q.query_id:{doc for doc, grade in grades[q.query_id].items() if grade > 0}
                for q in queries}

    results = []
    run_rows = []
    for query in queries:
        qt = Counter(tokens(query.text))
        scores = {}
        scores['TF-IDF'] = (matrix @ tfidf.transform([query.text]).T).toarray().ravel()
        bm = np.zeros(len(docs), dtype=float)
        lm = np.zeros(len(docs), dtype=float)
        for word, qtf in qt.items():
            df = docfreq[word]
            if df:
                idf = math.log(1 + (len(docs)-df+0.5)/(df+0.5))
                for i, tf in postings[word]:
                    bm[i] += qtf * idf * tf * 2.2 / (tf + 1.2*(0.25 + 0.75*lengths[i]/avgdl))
            # Dirichlet-smoothed multinomial query-likelihood score.
            mu = 1000.0
            pc = (vocab_counts[word] + 1e-12)/collection_len
            lm += qtf * np.log(mu * pc/(lengths + mu))
            for i, tf in postings[word]:
                lm[i] += qtf * math.log1p(tf/(mu*pc))
        scores['BM25'] = bm
        scores['LM-Dirichlet'] = lm
        for model, values in scores.items():
            order = sorted(range(len(docs)), key=lambda i:(-values[i], ids[i]))[:K]
            ranking = [ids[i] for i in order]
            metric = evaluate(ranking, relevant[query.query_id])
            metric['nDCG10'] = ndcg_at_k(ranking, grades[query.query_id], K)
            results.append(dict(Model=model,Query=query.query_id,**metric))
            for rank,i in enumerate(order,1):
                run_rows.append(dict(Model=model,Query=query.query_id,Rank=rank,
                                     DocID=ids[i],Score=f'{values[i]:.9f}',
                                     Relevance=grades[query.query_id].get(ids[i],0)))
    with (OUT/'part_c_per_query.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    with (OUT/'part_c_ranked_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(run_rows[0])); w.writeheader(); w.writerows(run_rows)
    agg = {}
    for model in ('TF-IDF','BM25','LM-Dirichlet'):
        group=[r for r in results if r['Model']==model]
        agg[model]={m:mean(r[m] for r in group)
                    for m in ('Precision','Recall','F1','P3','AP','RR','nDCG10')}
        agg[model]['MAP@10']=agg[model].pop('AP')
        agg[model]['MRR@10']=agg[model].pop('RR')
    metadata={'dataset':'cranfield','documents':len(docs),'queries':len(queries),
              'qrels':len(qrels),'top_k':K,'positive_grade_threshold':1,
              'tfidf':'scikit-learn L2 cosine, smooth IDF, unigrams',
              'bm25':'k1=1.2, b=0.75, Robertson probabilistic IDF',
              'language_model':'query likelihood, Dirichlet mu=1000',
              'metrics':'P/R/F1 at 10, P@3, AP@10, RR@10, nDCG@10; macro means'}
    (OUT/'part_c_summary.json').write_text(json.dumps({'metadata':metadata,'aggregate':agg},indent=2))
    print(json.dumps({'metadata':metadata,'aggregate':agg},indent=2))


if __name__=='__main__': main()
