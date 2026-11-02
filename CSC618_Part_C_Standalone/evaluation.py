"""Reusable information retrieval evaluation metrics; no third-party dependencies."""
import math


def precision_recall_f1(ranking, relevant):
    hits = sum(doc in relevant for doc in ranking)
    precision = hits / len(ranking) if ranking else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def precision_at_k(ranking, relevant, k=3):
    return sum(doc in relevant for doc in ranking[:k]) / k


def reciprocal_rank(ranking, relevant):
    return next((1 / i for i, doc in enumerate(ranking, 1) if doc in relevant), 0.0)


def average_precision(ranking, relevant):
    if not relevant:
        return 0.0
    return sum(precision_at_k(ranking, relevant, i)
               for i, doc in enumerate(ranking, 1) if doc in relevant) / len(relevant)


def dcg_at_k(ranking, grades, k=5):
    return sum((2 ** grades.get(doc, 0) - 1) / math.log2(i + 1)
               for i, doc in enumerate(ranking[:k], 1))


def ndcg_at_k(ranking, grades, k=5):
    ideal = sorted(grades, key=lambda d: grades[d], reverse=True)
    denominator = dcg_at_k(ideal, grades, k)
    return dcg_at_k(ranking, grades, k) / denominator if denominator else 0.0


def evaluate(ranking, relevant):
    p, r, f1 = precision_recall_f1(ranking, relevant)
    return dict(Precision=p, Recall=r, F1=f1,
                P3=precision_at_k(ranking, relevant, 3),
                AP=average_precision(ranking, relevant),
                RR=reciprocal_rank(ranking, relevant))
