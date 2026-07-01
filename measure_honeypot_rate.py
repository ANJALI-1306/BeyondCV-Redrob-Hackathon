import json
from typing import List, Dict, Any
from rank import load_candidates, rank_candidates
from features import honeypot_flag


def measure_honeypot_rate(ranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    honeypot_count = 0
    honeypot_positions = []

    for i, item in enumerate(ranked_candidates):
        rank = i + 1
        features = item.get('features', {})

        is_honeypot = features.get('honeypot_flag', False)

        if is_honeypot:
            honeypot_count += 1
            honeypot_positions.append(rank)

    honeypot_rate = honeypot_count / len(ranked_candidates) if ranked_candidates else 0

    return {
        'total_candidates': len(ranked_candidates),
        'honeypot_count': honeypot_count,
        'honeypot_rate': honeypot_rate,
        'honeypot_positions': honeypot_positions,
        'compliance': honeypot_rate <= 0.10,
    }


def detailed_honeypot_analysis(ranked_candidates: List[Dict[str, Any]]) -> str:
    stats = measure_honeypot_rate(ranked_candidates)

    report = []
    report.append("=" * 60)
    report.append("Honeypot Analysis Report")
    report.append("=" * 60)
    report.append(f"\nTotal candidates analyzed: {stats['total_candidates']}")
    report.append(f"Honeypot candidates found: {stats['honeypot_count']}")
    report.append(f"Honeypot rate: {stats['honeypot_rate']:.2%}")
    report.append(f"Compliance (<10%): {'✅ PASS' if stats['compliance'] else '❌ FAIL'}")

    if stats['honeypot_positions']:
        report.append(f"\nHoneypot positions in ranking:")
        for pos in stats['honeypot_positions']:
            report.append(f"  Rank {pos}: {ranked_candidates[pos-1]['candidate_id']}")
            report.append(f"    Score: {ranked_candidates[pos-1]['score']:.6f}")
            report.append(f"    Anomaly score: {ranked_candidates[pos-1]['features'].get('honeypot_anomaly_score', 0):.3f}")
    else:
        report.append("\nNo honeypots found in ranking.")

    report.append("\n" + "=" * 60)

    return "\n".join(report)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Measure honeypot rate in ranking')
    parser.add_argument('--ranking', default='ranking.csv', help='Path to ranking CSV')
    parser.add_argument('--candidates', help='Path to candidates.jsonl (if not using ranking.csv)')

    args = parser.parse_args()

    if args.candidates:
        print("Loading candidates...")
        candidates = load_candidates(args.candidates)

        print("Ranking candidates...")
        ranked = rank_candidates(candidates)
    else:
        print("Loading ranking from CSV...")
        import csv
        ranked = []
        with open(args.ranking, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ranked.append({
                    'candidate_id': row['candidate_id'],
                    'rank': int(row['rank']),
                    'score': float(row['score']),
                    'reasoning': row['reasoning'],
                    'features': {}
                })

        print("Note: Full honeypot analysis requires candidate features.")
        print("Use --candidates to analyze from source.")

    if args.candidates:
        report = detailed_honeypot_analysis(ranked)
        print(report)
    else:
        print("Full honeypot analysis requires --candidates parameter.")
