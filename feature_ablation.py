import json
import numpy as np
from typing import Dict, Any, List
from rank import load_candidates, rank_candidates
from evaluation import evaluate_ranking, generate_evaluation_report
from features import extract_features


class FeatureAblator:

    def __init__(self):

        self.feature_groups = {
            'semantic': [
                'semantic_similarity',
                'title_semantic_similarity',
                'skill_jd_overlap',
                'career_description_semantic_similarity',
                'company_name_semantic_similarity',
                'skill_name_semantic_similarity',
                'education_field_semantic_similarity',
            ],
            'experience': [
                'total_years',
                'relevant_years',
                'seniority_level',
                'domain_experience',
                'management_experience',
            ],
            'skills': [
                'skill_count',
                'expert_skill_count',
                'skill_diversity',
                'skill_recency',
                'rare_skill_score',
            ],
            'career': [
                'promotion_frequency',
                'average_tenure',
                'career_growth_velocity',
                'career_consistency',
                'employer_quality',
            ],
            'education': [
                'education_tier_score',
                'field_relevance',
                'education_count',
                'highest_degree',
                'graduation_recency',
            ],
            'behavioral': [
                'profile_completeness',
                'response_rate',
                'response_time',
                'activity_score',
                'search_appearance',
                'saved_by_recruiters',
            ],
            'redrob_extended': [
                'applications_30d',
                'profile_views_30d',
                'connection_count',
                'endorsements_received',
                'github_activity',
                'interview_completion_rate',
                'offer_acceptance_rate',
                'notice_period_alignment',
                'salary_range_alignment',
                'work_mode_alignment',
                'willing_to_relocate',
                'verified_email',
                'verified_phone',
                'linkedin_connected',
            ],
            'disqualifiers': [
                'disqualifier_penalty',
            ],
            'honeypot': [
                'honeypot_flag',
                'honeypot_penalty_multiplier',
                'honeypot_anomaly_score',
            ],
        }
    
    def ablate_features(self, features: Dict[str, float],
                        group_to_remove: str) -> Dict[str, float]:

        ablated = features.copy()

        if group_to_remove in self.feature_groups:
            for feature in self.feature_groups[group_to_remove]:
                if feature in ablated:
                    ablated[feature] = 0.0

        return ablated

    def run_ablation_study(self, candidates: List[Dict[str, Any]],
                          validation_labels: Dict[str, int] = None,
                          validation_file: str = None) -> Dict[str, Any]:

        print("Running baseline ranking...")
        ranked_baseline = rank_candidates(candidates)

        if validation_labels or validation_file:
            baseline_metrics = evaluate_ranking(
                ranked_baseline, validation_labels, validation_file
            )
        else:
            baseline_metrics = {'competition_score': 0.0}

        results = {
            'baseline': baseline_metrics,
            'ablations': {}
        }

        for group_name in self.feature_groups:
            print(f"\nAblating {group_name}...")

            ranked_ablated = self._rank_with_ablation(
                candidates, group_name
            )

            if validation_labels or validation_file:
                ablated_metrics = evaluate_ranking(
                    ranked_ablated, validation_labels, validation_file
                )
            else:
                ablated_metrics = {'competition_score': 0.0}

            impact = baseline_metrics.get('competition_score', 0) - ablated_metrics.get('competition_score', 0)

            results['ablations'][group_name] = {
                'metrics': ablated_metrics,
                'impact': impact,
                'impact_pct': (impact / baseline_metrics.get('competition_score', 1)) * 100 if baseline_metrics.get('competition_score', 0) > 0 else 0
            }

            print(f"  Impact: {impact:.4f} ({results['ablations'][group_name]['impact_pct']:.1f}%)")

        return results

    def _rank_with_ablation(self, candidates: List[Dict[str, Any]],
                          group_to_remove: str) -> List[Dict[str, Any]]:

        from scoring import compute_score
        from reasoning import generate_reasoning

        ranked = []

        for candidate in candidates:

            features = extract_features(candidate)

            ablated_features = self.ablate_features(features, group_to_remove)

            score = compute_score(ablated_features, candidate['candidate_id'])

            ranked.append({
                'candidate_id': candidate['candidate_id'],
                'score': score,
                'features': ablated_features,
                'candidate': candidate
            })

        ranked.sort(key=lambda x: (-x['score'], x['candidate_id']))

        for idx, item in enumerate(ranked[:100]):
            rank = idx + 1
            item['rank'] = rank

            if rank <= 33:
                tier = 'high'
            elif rank <= 66:
                tier = 'medium'
            else:
                tier = 'low'

            reasoning = generate_reasoning(
                item['features'],
                item['score'],
                item['candidate'],
                tier
            )
            item['reasoning'] = reasoning

        return ranked[:100]

    def generate_ablation_report(self, results: Dict[str, Any]) -> str:

        report = []
        report.append("=" * 60)
        report.append("Feature Ablation Study Report")
        report.append("=" * 60)

        baseline = results['baseline']
        report.append(f"\nBaseline Competition Score: {baseline.get('competition_score', 0):.4f}")

        report.append("\nFeature Group Impact Analysis:")
        report.append("-" * 60)
        report.append(f"{'Group':<20} {'Impact':>10} {'Impact %':>10}")
        report.append("-" * 60)

        sorted_ablations = sorted(
            results['ablations'].items(),
            key=lambda x: x[1]['impact'],
            reverse=True
        )

        for group_name, ablation in sorted_ablations:
            impact = ablation['impact']
            impact_pct = ablation['impact_pct']
            report.append(f"{group_name:<20} {impact:>10.4f} {impact_pct:>9.1f}%")

        report.append("-" * 60)

        report.append("\nSummary:")
        most_important = sorted_ablations[0] if sorted_ablations else None
        if most_important:
            report.append(f"  Most important feature group: {most_important[0]}")
            report.append(f"  Impact: {most_important[1]['impact']:.4f} ({most_important[1]['impact_pct']:.1f}%)")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run feature ablation study')
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl')
    parser.add_argument('--validation', help='Path to validation_labeled.json')
    parser.add_argument('--output', default='ablation_report.txt', help='Output report file')

    args = parser.parse_args()

    print("Loading candidates...")
    candidates = load_candidates(args.candidates)

    print("Running ablation study...")
    ablator = FeatureAblator()
    results = ablator.run_ablation_study(
        candidates,
        validation_file=args.validation
    )

    print("\nGenerating report...")
    report = ablator.generate_ablation_report(results)
    print(report)

    with open(args.output, 'w') as f:
        f.write(report)

    print(f"\nReport saved to {args.output}")
