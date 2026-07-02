from typing import Dict, Any, List
import os
import hashlib


def compute_score_manual(features: Dict[str, Any], candidate_id: str = "") -> float:

    w_title = 0.25
    w_skill = 0.20
    w_production = 0.15
    w_experience = 0.10
    w_location = 0.01

    w_semantic = 0.10
    w_skill_overlap = 0.05

    w_domain = 0.05
    w_employer = 0.03
    w_engagement = 0.03
    w_rare_skill = 0.02
    w_skill_div = 0.02
    w_career_cons = 0.02

    title_role_fit = features['title_role_fit_score']
    experience_fit = features['experience_fit_score']
    skill_coherence = features['skill_coherence_score']
    production_seniority = features['production_seniority_score']
    location_fit = features['location_fit_score']
    disqualifier_penalty = features['disqualifier_penalty']
    behavioral_multiplier = features['behavioral_multiplier']
    honeypot_penalty_multiplier = features['honeypot_penalty_multiplier']

    semantic_similarity = features.get('semantic_similarity', 0.0)
    skill_jd_overlap = features.get('skill_jd_overlap', 0.0)

    domain_experience = features.get('domain_experience', 0.0)
    employer_quality = features.get('employer_quality', 0.0)
    engagement_score = features.get('engagement_score', 0.0)
    rare_skill_score = features.get('rare_skill_score', 0.0)
    skill_diversity = features.get('skill_diversity', 0.0)
    career_consistency = features.get('career_consistency', 0.0)

    base = (
        w_title * title_role_fit +
        w_skill * skill_coherence +
        w_production * production_seniority +
        w_experience * experience_fit +
        w_location * location_fit +
        w_semantic * semantic_similarity +
        w_skill_overlap * skill_jd_overlap +
        w_domain * domain_experience +
        w_employer * employer_quality +
        w_engagement * engagement_score +
        w_rare_skill * rare_skill_score +
        w_skill_div * skill_diversity +
        w_career_cons * career_consistency +
        disqualifier_penalty
    )

    base_clipped = max(min(base, 1.0), 0.0)

    if candidate_id:
        
        
        digest = hashlib.md5(candidate_id.encode("utf-8")).hexdigest()
        stable_hash = int(digest[:8], 16)
        tiebreaker = (stable_hash % 10000) * 0.000001
        base_clipped += tiebreaker

    base_clipped = max(min(base_clipped, 1.0), 0.0)

    final_score = base_clipped * behavioral_multiplier * honeypot_penalty_multiplier

    return final_score


def compute_score_ltr(features: Dict[str, Any], model, feature_names: List[str]) -> float:

    import numpy as np

    feature_vector = np.array([features.get(name, 0.0) for name in feature_names])

    feature_vector = feature_vector.reshape(1, -1)

    score = model.predict(feature_vector)[0]

    score_normalized = 1.0 / (1.0 + np.exp(-score))

    return float(score_normalized)


def compute_score(features: Dict[str, Any], candidate_id: str = "", model=None, feature_names=None) -> float:

    if model is not None and feature_names is not None:
        try:
            return compute_score_ltr(features, model, feature_names)
        except Exception as e:
            print(f"Warning: LTR prediction failed, falling back to manual weights: {e}")

    return compute_score_manual(features, candidate_id)
