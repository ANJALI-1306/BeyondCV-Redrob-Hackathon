import json
from typing import Dict, Any


def heuristic_relevance_score(candidate: Dict[str, Any]) -> int:

    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    signals = candidate.get('redrob_signals', {})

    score = 0

    years = profile.get('years_of_experience', 0)
    if 5 <= years <= 9:
        exp_score = 1.0
    elif 3 <= years < 5 or 9 < years <= 12:
        exp_score = 0.7
    elif years >= 2:
        exp_score = 0.4
    else:
        exp_score = 0.0

    title = profile.get('current_title', '').lower()
    ai_ml_keywords = ['machine learning', 'ai engineer', 'ml engineer', 'data scientist',
                      'research scientist', 'nlp engineer', 'senior ml', 'staff ml', 'principal ml']
    if any(kw in title for kw in ai_ml_keywords):
        title_score = 1.0
    elif 'engineer' in title or 'developer' in title:
        title_score = 0.5
    else:
        title_score = 0.0
    consulting_keywords = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini']
    product_score = 1.0
    for role in career:
        company = role.get('company', '').lower()
        if any(cons in company for cons in consulting_keywords):
            product_score = 0.3
            break

    skill_names = [s.get('name', '').lower() for s in skills]
    required_skills = ['pytorch', 'tensorflow', 'transformer', 'rag', 'pinecone',
                      'weaviate', 'qdrant', 'vector', 'nlp', 'machine learning']
    skill_count = sum(1 for skill in skill_names if any(req in skill for req in required_skills))
    skill_score = min(skill_count / 5, 1.0)

    response_rate = signals.get('recruiter_response_rate', 0)
    last_active = signals.get('last_active_date', '')
    open_to_work = signals.get('open_to_work_flag', False)

    if response_rate > 0.7 and open_to_work:
        behavioral_score = 1.0
    elif response_rate > 0.5:
        behavioral_score = 0.7
    elif response_rate > 0.3:
        behavioral_score = 0.4
    else:
        behavioral_score = 0.0

    location = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
    india_locations = ['bangalore', 'hyderabad', 'pune', 'mumbai', 'delhi', 'noida', 'ncr']
    willing_to_relocate = signals.get('willing_to_relocate', False)

    if country == 'india' or any(loc in location for loc in india_locations):
        location_score = 1.0
    elif willing_to_relocate:
        location_score = 0.7
    else:
        location_score = 0.3

    weights = {
        'experience': 0.25,
        'title': 0.25,
        'product': 0.15,
        'skills': 0.2,
        'behavioral': 0.1,
        'location': 0.05,
    }

    composite = (
        weights['experience'] * exp_score +
        weights['title'] * title_score +
        weights['product'] * product_score +
        weights['skills'] * skill_score +
        weights['behavioral'] * behavioral_score +
        weights['location'] * location_score
    )

    if composite >= 0.8:
        return 4
    elif composite >= 0.6:
        return 3
    elif composite >= 0.4:
        return 2
    elif composite >= 0.2:
        return 1
    else:
        return 0


def create_labeled_dataset(template_file: str, output_file: str):

    with open(template_file, 'r', encoding='utf-8') as f:
        template = json.load(f)

    labeled = []
    label_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for entry in template:

        candidate = {
            'candidate_id': entry['candidate_id'],
            'profile': {
                'current_title': entry['current_title'],
                'current_company': entry['current_company'],
                'years_of_experience': entry['years_of_experience'],
                'location': entry['location'],
                'headline': entry['headline'],
                'summary': entry['summary'],
            },
            'career_history': entry['recent_career'],
            'skills': entry['top_skills'],
            'education': entry['education'],
            'redrob_signals': entry['behavioral'],
        }

        relevance = heuristic_relevance_score(candidate)
        entry['relevance_label'] = relevance
        entry['labeling_notes'] = f'Heuristic label based on JD requirements (composite score: {relevance})'

        label_distribution[relevance] += 1
        labeled.append(entry)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labeled, f, indent=2, ensure_ascii=False)

    print(f"Labeled dataset created: {output_file}")
    print(f"Label distribution:")
    for label, count in sorted(label_distribution.items()):
        print(f"  Relevance {label}: {count} candidates ({count/len(labeled)*100:.1f}%)")

    return labeled


def load_validation_labels(validation_file: str) -> Dict[str, int]:

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

    parser = argparse.ArgumentParser(description='Create heuristic validation labels')
    parser.add_argument('--template', default='validation_template.json', help='Input template file')
    parser.add_argument('--output', default='validation_labeled.json', help='Output labeled file')

    args = parser.parse_args()

    create_labeled_dataset(args.template, args.output)
