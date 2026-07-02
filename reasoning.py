
from typing import Dict, Any


def generate_reasoning(features: Dict[str, Any], score: float, candidate: Dict[str, Any] = None, tier: str = None) -> str:

    if not candidate:
        return "Candidate profile data unavailable for reasoning generation."

    profile = candidate.get('profile', {})
    career_history = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    redrob_signals = candidate.get('redrob_signals', {})

    current_title = profile.get('current_title', 'Unknown')
    current_company = profile.get('current_company', 'Unknown')
    years_of_experience = profile.get('years_of_experience', 0)
    location = profile.get('location', 'Unknown')
    country = profile.get('country', 'Unknown')
    willing_to_relocate = redrob_signals.get('willing_to_relocate', False)

    title_fit = features['title_role_fit_score']
    exp_fit = features['experience_fit_score']
    skill_coherence = features['skill_coherence_score']
    production_seniority = features['production_seniority_score']
    location_fit = features['location_fit_score']
    disqualifier_penalty = features['disqualifier_penalty']
    behavioral_multiplier = features['behavioral_multiplier']
    honeypot_flag = features['honeypot_flag']

    semantic_similarity = features.get('semantic_similarity', 0.0)
    domain_experience = features.get('domain_experience', 0.0)
    employer_quality = features.get('employer_quality', 0.0)
    engagement_score = features.get('engagement_score', 0.0)
    rare_skill_score = features.get('rare_skill_score', 0.0)

    if tier is None:
        if score >= 0.7:
            tier = 'high'
        elif score >= 0.4:
            tier = 'medium'
        else:
            tier = 'low'

    parts = []

    parts.append(f"{current_title} at {current_company} with {years_of_experience} years of experience")

    top_skills = []
    for skill in skills[:5]:
        if skill.get('proficiency') in ['expert', 'advanced']:
            top_skills.append(skill.get('name'))
    if top_skills:
        parts.append(f"demonstrates expertise in {', '.join(top_skills[:3])}")

    if skill_coherence >= 0.7:
        parts.append("with strong skill coherence and career evidence")
    elif skill_coherence < 0.4:
        parts.append("with limited skill coherence evidence")

    if production_seniority >= 0.7:
        parts.append("and proven production deployment capabilities")
    elif production_seniority < 0.4:
        parts.append("with limited production deployment evidence")

    if semantic_similarity >= 0.6:
        parts.append("showing strong contextual alignment with JD requirements")
    elif semantic_similarity < 0.3:
        parts.append("with weak contextual match to job requirements")

    if tier == 'high':
        if domain_experience >= 0.7:
            parts.append("in relevant AI/ML domains")
        if employer_quality >= 0.7:
            parts.append("at high-quality product companies")
        if engagement_score >= 0.8:
            parts.append("with strong platform engagement")
    elif tier == 'medium':
        if domain_experience < 0.5:
            parts.append("with limited domain-specific experience")
        if employer_quality < 0.5:
            parts.append("primarily at consulting/service companies")
        if behavioral_multiplier < 0.7:
            parts.append("and recent inactivity affecting availability")
    else:
        if title_fit < 0.5:
            parts.append("showing limited alignment with Senior AI Engineer role")
        if exp_fit < 0.4:
            parts.append(f"with experience outside the target 5-9 year band")
        if honeypot_flag:
            parts.append("and profile inconsistencies requiring verification")
        if behavioral_multiplier < 0.5:
            parts.append("with concerning platform inactivity patterns")

    if location_fit >= 0.7:
        parts.append(f"based in {location}")
    elif location_fit < 0.6:
        if willing_to_relocate:
            parts.append(f"currently in {location} but willing to relocate")
        else:
            parts.append(f"based in {location} with relocation concerns")

    reasoning = '; '.join(parts) + '.'

    if len(reasoning) > 300:
        reasoning = reasoning[:297] + '...'

    return reasoning
