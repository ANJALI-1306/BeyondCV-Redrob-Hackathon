import numpy as np
from typing import List, Dict, Any
import math
import json


def dcg_at_k(relevance_scores: List[int], k: int) -> float:

    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        relevance = relevance_scores[i]
        if relevance > 0:
            dcg += relevance / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevance_scores: List[int], k: int) -> float:

    dcg = dcg_at_k(relevance_scores, k)

    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)

    if idcg > 0:
        return dcg / idcg
    else:
        return 0.0


def average_precision(relevance_scores: List[int]) -> float:

    num_relevant = sum(1 for score in relevance_scores if score > 0)

    if num_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, score in enumerate(relevance_scores):
        if score > 0:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    return precision_sum / num_relevant


def precision_at_k(relevance_scores: List[int], k: int) -> float:

    k = min(k, len(relevance_scores))
    relevant_count = sum(1 for score in relevance_scores[:k] if score > 0)
    return relevant_count / k if k > 0 else 0.0


def evaluate_ranking(ranked_candidates: List[Dict[str, Any]],
                     true_relevance: Dict[str, int] = None,
                     validation_file: str = None) -> Dict[str, float]:

    if validation_file and not true_relevance:
        with open(validation_file, 'r', encoding='utf-8') as f:
            validation_data = json.load(f)
        true_relevance = {}
        for entry in validation_data:
            if entry.get('relevance_label') is not None:
                true_relevance[entry['candidate_id']] = entry['relevance_label']

    relevance_scores = []
    for item in ranked_candidates:
        candidate_id = item['candidate_id']
        relevance_scores.append(true_relevance.get(candidate_id, 0))

    metrics = {
        'ndcg@10': ndcg_at_k(relevance_scores, 10),
        'ndcg@50': ndcg_at_k(relevance_scores, 50),
        'map': average_precision(relevance_scores),
        'precision@10': precision_at_k(relevance_scores, 10),
    }

    metrics['competition_score'] = (
        0.50 * metrics['ndcg@10'] +
        0.30 * metrics['ndcg@50'] +
        0.15 * metrics['map'] +
        0.05 * metrics['precision@10']
    )

    return metrics


def generate_pseudo_relevance(ranked_candidates: List[Dict[str, Any]]) -> Dict[str, int]:

    pseudo_relevance = {}

    for i, item in enumerate(ranked_candidates):
        candidate_id = item['candidate_id']
        rank = i + 1

        if rank <= 10:
            relevance = 4
        elif rank <= 30:
            relevance = 3
        elif rank <= 60:
            relevance = 2
        elif rank <= 100:
            relevance = 1
        else:
            relevance = 0

        pseudo_relevance[candidate_id] = relevance

    return pseudo_relevance


def ranking_diagnostics(ranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:

    scores = [item['score'] for item in ranked_candidates]

    diagnostics = {
        'num_candidates': len(ranked_candidates),
        'score_mean': np.mean(scores),
        'score_std': np.std(scores),
        'score_min': np.min(scores),
        'score_max': np.max(scores),
        'score_median': np.median(scores),
        'score_range': np.max(scores) - np.min(scores),
        'score_distribution': {
            'q1': np.percentile(scores, 25),
            'q2': np.percentile(scores, 50),
            'q3': np.percentile(scores, 75),
        }
    }

    return diagnostics


def generate_evaluation_report(ranked_candidates: List[Dict[str, Any]],
                              true_relevance: Dict[str, int] = None,
                              validation_file: str = None) -> str:

    report = []
    report.append("=" * 60)
    report.append("Ranking Evaluation Report")
    report.append("=" * 60)

    diagnostics = ranking_diagnostics(ranked_candidates)
    report.append("\n[Ranking Diagnostics]")
    report.append(f"  Number of candidates: {diagnostics['num_candidates']}")
    report.append(f"  Score mean: {diagnostics['score_mean']:.6f}")
    report.append(f"  Score std: {diagnostics['score_std']:.6f}")
    report.append(f"  Score min: {diagnostics['score_min']:.6f}")
    report.append(f"  Score max: {diagnostics['score_max']:.6f}")
    report.append(f"  Score median: {diagnostics['score_median']:.6f}")
    report.append(f"  Score range: {diagnostics['score_range']:.6f}")
    report.append(f"  Score Q1: {diagnostics['score_distribution']['q1']:.6f}")
    report.append(f"  Score Q2: {diagnostics['score_distribution']['q2']:.6f}")
    report.append(f"  Score Q3: {diagnostics['score_distribution']['q3']:.6f}")

    if true_relevance or validation_file:
        metrics = evaluate_ranking(ranked_candidates, true_relevance, validation_file)
        report.append("\n[Evaluation Metrics]")
        report.append(f"  NDCG@10: {metrics['ndcg@10']:.6f}")
        report.append(f"  NDCG@50: {metrics['ndcg@50']:.6f}")
        report.append(f"  MAP: {metrics['map']:.6f}")
        report.append(f"  Precision@10: {metrics['precision@10']:.6f}")
        report.append(f"  Competition Score: {metrics['competition_score']:.6f}")
        if validation_file:
            report.append("  Note: Using heuristic validation labels from validation_labeled.json")
    else:
        pseudo_relevance = generate_pseudo_relevance(ranked_candidates)
        metrics = evaluate_ranking(ranked_candidates, pseudo_relevance)
        report.append("\n[Pseudo-Evaluation Metrics (no ground truth)]")
        report.append(f"  NDCG@10: {metrics['ndcg@10']:.6f}")
        report.append(f"  NDCG@50: {metrics['ndcg@50']:.6f}")
        report.append(f"  MAP: {metrics['map']:.6f}")
        report.append(f"  Precision@10: {metrics['precision@10']:.6f}")
        report.append("  Note: These are based on pseudo-labels from ranking position")

    report.append("\n[Top 10 Candidates]")
    for i, item in enumerate(ranked_candidates[:10]):
        report.append(f"  {i+1}. {item['candidate_id']} - Score: {item['score']:.6f}")
        report.append(f"      Reasoning: {item['reasoning'][:80]}...")

    report.append("\n" + "=" * 60)

    return "\n".join(report)


if __name__ == '__main__':
    from rank import load_candidates, rank_candidates

    print("Loading candidates...")
    candidates_file = r'c:\Users\anjal\Downloads\dataset_extracted\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'
    candidates = load_candidates(candidates_file)

    print("Ranking candidates...")
    ranked = rank_candidates(candidates)

    print("\nGenerating evaluation report...")
    report = generate_evaluation_report(ranked, validation_file='validation_labeled.json')
    print(report)
