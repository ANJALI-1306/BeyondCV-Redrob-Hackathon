import json
import random
from typing import List, Dict, Any
from pathlib import Path


def sample_representative_candidates(candidates_file: str, n: int = 200) -> List[Dict[str, Any]]:
    candidates = []
    with open(candidates_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    print(f"Total candidates in dataset: {len(candidates)}")

    ai_ml_keywords = ['machine learning', 'ai engineer', 'ml engineer', 'data scientist',
                      'research scientist', 'nlp engineer', 'computer vision']
    adjacent_keywords = ['software engineer', 'backend engineer', 'frontend engineer',
                        'full stack', 'data engineer', 'analytics engineer']

    categories = {
        'ai_ml_senior': [],
        'ai_ml_mid': [],
        'ai_ml_junior': [],
        'adjacent_senior': [],
        'adjacent_mid': [],
        'adjacent_junior': [],
        'unrelated': [],
    }

    for candidate in candidates:
        profile = candidate.get('profile', {})
        title = profile.get('current_title', '').lower()
        years = profile.get('years_of_experience', 0)

        if any(kw in title for kw in ai_ml_keywords):
            if years >= 5:
                categories['ai_ml_senior'].append(candidate)
            elif years >= 3:
                categories['ai_ml_mid'].append(candidate)
            else:
                categories['ai_ml_junior'].append(candidate)
        elif any(kw in title for kw in adjacent_keywords):
            if years >= 5:
                categories['adjacent_senior'].append(candidate)
            elif years >= 3:
                categories['adjacent_mid'].append(candidate)
            else:
                categories['adjacent_junior'].append(candidate)
        else:
            categories['unrelated'].append(candidate)

    sampled = []
    for category, cat_candidates in categories.items():
        if cat_candidates:
            sample_size = max(10, min(len(cat_candidates), n // 7))
            sampled.extend(random.sample(cat_candidates, min(sample_size, len(cat_candidates))))

    if len(sampled) < n:
        remaining = [c for c in candidates if c not in sampled]
        needed = n - len(sampled)
        sampled.extend(random.sample(remaining, min(needed, len(remaining))))

    print(f"Sampled {len(sampled)} candidates from categories:")
    for category, cat_candidates in categories.items():
        count = sum(1 for c in sampled if c in cat_candidates)
        print(f"  {category}: {count}")

    return sampled[:n]


def create_validation_template(sampled_candidates: List[Dict[str, Any]], output_file: str):
    template = []

    for candidate in sampled_candidates:
        profile = candidate.get('profile', {})
        career = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        education = candidate.get('education', [])
        signals = candidate.get('redrob_signals', {})

        entry = {
            'candidate_id': candidate['candidate_id'],
            'current_title': profile.get('current_title', ''),
            'current_company': profile.get('current_company', ''),
            'years_of_experience': profile.get('years_of_experience', 0),
            'location': profile.get('location', ''),
            'headline': profile.get('headline', ''),
            'summary': profile.get('summary', ''),
            'top_skills': [],
            'recent_career': [],
            'education': [],
            'behavioral': {},
            'relevance_label': None,
            'labeling_notes': '',
        }

        sorted_skills = sorted(skills, key=lambda s: (s.get('endorsements', 0),
                                                      {'expert': 4, 'advanced': 3,
                                                       'intermediate': 2, 'beginner': 1}.get(s.get('proficiency', 'beginner'), 1)),
                              reverse=True)
        entry['top_skills'] = sorted_skills[:5]

        entry['recent_career'] = career[:3]

        tier_order = {'tier_1': 1, 'tier_2': 2, 'tier_3': 3, 'tier_4': 4, 'unknown': 5}
        sorted_edu = sorted(education, key=lambda e: tier_order.get(e.get('tier', 'unknown'), 5))
        entry['education'] = sorted_edu

        entry['behavioral'] = {
            'response_rate': signals.get('recruiter_response_rate', 0),
            'avg_response_time_hours': signals.get('avg_response_time_hours', 0),
            'last_active_date': signals.get('last_active_date', ''),
            'profile_views_30d': signals.get('profile_views_received_30d', 0),
            'applications_30d': signals.get('applications_submitted_30d', 0),
            'open_to_work': signals.get('open_to_work_flag', False),
            'github_activity': signals.get('github_activity_score', -1),
        }

        template.append(entry)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"Validation template created: {output_file}")
    print(f"Contains {len(template)} candidates for manual labeling")
    print(f"\nLabeling instructions:")
    print("  relevance_label: 0 (not relevant), 1 (marginally relevant), 2 (somewhat relevant),")
    print("                   3 (relevant), 4 (highly relevant)")
    print("  Base labels on JD requirements:")
    print("    - AI/ML experience (5-9 years preferred)")
    print("    - Product company experience")
    print("    - Production ML systems")
    print("    - NLP/RAG/vector databases")
    print("    - Behavioral engagement")
    print("    - Location preference (India)")
    print("  labeling_notes: Brief justification for the label")


def load_validation_dataset(validation_file: str) -> Dict[str, int]:
    with open(validation_file, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    labels = {}
    for entry in validation_data:
        if entry.get('relevance_label') is not None:
            labels[entry['candidate_id']] = entry['relevance_label']

    print(f"Loaded {len(labels)} labeled candidates from validation dataset")
    return labels


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build validation dataset for manual labeling')
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl')
    parser.add_argument('--output', default='validation_template.json', help='Output template file')
    parser.add_argument('--n', type=int, default=200, help='Number of candidates to sample')

    args = parser.parse_args()

    sampled = sample_representative_candidates(args.candidates, args.n)

    create_validation_template(sampled, args.output)

    print("\nNext steps:")
    print("1. Manually review validation_template.json")
    print("2. Assign relevance_label (0-4) to each candidate")
    print("3. Add labeling_notes for justification")
    print("4. Save as validation_labeled.json")
    print("5. Use with evaluation.py for proper evaluation")
