from typing import Dict, Any, List, Optional
import re


def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    if dictionary is None:
        return default
    return dictionary.get(key, default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def normalize_string(text: str) -> str:
    if text is None:
        return ""
    return text.lower().strip()


def validate_candidate_id(candidate_id: str) -> bool:
    if not candidate_id:
        return False
    pattern = r'^CAND_\d+$'
    return bool(re.match(pattern, candidate_id))


def validate_date_string(date_str: str) -> bool:
    if not date_str:
        return False
    try:
        from datetime import datetime
        datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def deduplicate_list(items: List[Any]) -> List[Any]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def validate_profile_completeness(profile: Dict[str, Any]) -> Dict[str, bool]:
    required_fields = ['current_title', 'current_company', 'years_of_experience', 'location']
    return {
        field: bool(safe_get(profile, field))
        for field in required_fields
    }


def handle_missing_skills(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not skills:
        return []

    cleaned_skills = []
    for skill in skills:
        if not isinstance(skill, dict):
            continue

        cleaned_skill = {
            'name': safe_string(skill.get('name')),
            'proficiency': safe_string(skill.get('proficiency', 'beginner')),
            'duration_months': safe_int(skill.get('duration_months', 0)),
            'endorsements': safe_int(skill.get('endorsements', 0)),
        }

        if cleaned_skill['name']:
            cleaned_skills.append(cleaned_skill)

    return cleaned_skills


def handle_missing_career_history(career_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not career_history:
        return []

    cleaned_history = []
    for role in career_history:
        if not isinstance(role, dict):
            continue

        cleaned_role = {
            'title': safe_string(role.get('title')),
            'company': safe_string(role.get('company')),
            'duration_months': safe_int(role.get('duration_months', 0)),
            'description': clean_text(role.get('description', '')),
        }

        if cleaned_role['title']:
            cleaned_history.append(cleaned_role)

    return cleaned_history


def validate_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}

    candidate_id = safe_string(candidate.get('candidate_id'))
    if not validate_candidate_id(candidate_id):
        candidate_id = 'UNKNOWN'

    profile = candidate.get('profile', {})
    if not isinstance(profile, dict):
        profile = {}

    skills = handle_missing_skills(candidate.get('skills', []))

    career_history = handle_missing_career_history(candidate.get('career_history', []))

    education = candidate.get('education', [])
    if not isinstance(education, list):
        education = []

    redrob_signals = candidate.get('redrob_signals', {})
    if not isinstance(redrob_signals, dict):
        redrob_signals = {}

    return {
        'candidate_id': candidate_id,
        'profile': profile,
        'skills': skills,
        'career_history': career_history,
        'education': education,
        'redrob_signals': redrob_signals,
    }
