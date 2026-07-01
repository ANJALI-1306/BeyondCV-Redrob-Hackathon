"""
Unit tests for feature extraction using synthetic mini-profiles.

These tests use synthetic data to sanity-check feature logic without
needing to parse the full 100K candidate dataset.
"""

import pytest
from features import (
    title_role_fit_score,
    experience_fit_score,
    skill_coherence_score,
    production_seniority_score,
    location_fit_score,
    disqualifier_penalty,
    behavioral_multiplier,
    honeypot_flag,
    honeypot_penalty_multiplier,
    experience_features,
    skill_features,
    career_features,
    education_features,
    behavioral_features,
    resume_quality_features,
)
from reasoning import generate_reasoning
from scoring import compute_score


def test_title_role_fit_score_tier_a():
    """Test Tier A (core AI/ML roles) returns 1.0"""
    profile = {
        'current_title': 'ML Engineer',
        'years_of_experience': 6
    }
    career_history = [
        {'title': 'Data Scientist', 'company': 'Product Co', 'duration_months': 36},
        {'title': 'ML Engineer', 'company': 'Tech Corp', 'duration_months': 24}
    ]
    
    score = title_role_fit_score(profile, career_history)
    assert score == 1.0, f"Expected 1.0 for ML Engineer, got {score}"


def test_title_role_fit_score_tier_b():
    """Test Tier B (adjacent roles) returns 0.6"""
    profile = {
        'current_title': 'Data Engineer',
        'years_of_experience': 5
    }
    career_history = [
        {'title': 'Backend Engineer', 'company': 'Tech Corp', 'duration_months': 48}
    ]
    
    score = title_role_fit_score(profile, career_history)
    assert score == 0.6, f"Expected 0.6 for Data Engineer, got {score}"


def test_title_role_fit_score_tier_d():
    """Test Tier D (disqualifiers) returns 0.0"""
    profile = {
        'current_title': 'Marketing Manager',
        'years_of_experience': 8
    }
    career_history = [
        {'title': 'HR Manager', 'company': 'Services Co', 'duration_months': 36}
    ]
    
    score = title_role_fit_score(profile, career_history)
    assert score == 0.0, f"Expected 0.0 for Marketing Manager, got {score}"


def test_title_role_fit_score_nlp_engineer():
    """Test NLP Engineer scores Tier A (FIX 1 regression test)"""
    profile = {
        'current_title': 'NLP Engineer',
        'years_of_experience': 6
    }
    career_history = [
        {'title': 'NLP Engineer', 'company': 'AI Co', 'duration_months': 48},
        {'title': 'ML Engineer', 'company': 'Tech Corp', 'duration_months': 24}
    ]
    
    score = title_role_fit_score(profile, career_history)
    assert score == 1.0, f"Expected 1.0 for NLP Engineer, got {score}"


def test_title_role_fit_score_recommendation_systems_engineer():
    """Test Recommendation Systems Engineer scores Tier A (FIX 1 regression test)"""
    profile = {
        'current_title': 'Recommendation Systems Engineer',
        'years_of_experience': 5
    }
    career_history = [
        {'title': 'Recommendation Systems Engineer', 'company': 'Product Co', 'duration_months': 36}
    ]
    
    score = title_role_fit_score(profile, career_history)
    assert score == 1.0, f"Expected 1.0 for Recommendation Systems Engineer, got {score}"


def test_title_role_fit_score_senior_nlp_engineer():
    """Test Senior NLP Engineer scores Tier A (FIX 1 regression test)"""
    profile = {
        'current_title': 'Senior NLP Engineer',
        'years_of_experience': 7
    }
    career_history = [
        {'title': 'Senior NLP Engineer', 'company': 'AI Startup', 'duration_months': 48}
    ]
    
    score = title_role_fit_score(profile, career_history)
    assert score == 1.0, f"Expected 1.0 for Senior NLP Engineer, got {score}"


def test_experience_fit_score_ideal():
    """Test ideal experience (5-9 years) returns 1.0"""
    profile = {'years_of_experience': 7}
    career_history = []
    
    score = experience_fit_score(profile, career_history)
    assert score == 1.0, f"Expected 1.0 for 7 years, got {score}"


def test_experience_fit_score_close():
    """Test close to ideal (4 years) returns 0.8"""
    profile = {'years_of_experience': 4}
    career_history = []
    
    score = experience_fit_score(profile, career_history)
    assert score == 0.8, f"Expected 0.8 for 4 years, got {score}"


def test_experience_fit_score_low():
    """Test too low (2 years) returns 0.2"""
    profile = {'years_of_experience': 2}
    career_history = []
    
    score = experience_fit_score(profile, career_history)
    assert score == 0.2, f"Expected 0.2 for 2 years, got {score}"


def test_skill_coherence_score_high():
    """Test high coherence with career evidence"""
    skills = [
        {'name': 'Python', 'proficiency': 'expert', 'duration_months': 48, 'endorsements': 10},
        {'name': 'PyTorch', 'proficiency': 'advanced', 'duration_months': 24, 'endorsements': 5},
        {'name': 'Pinecone', 'proficiency': 'intermediate', 'duration_months': 12, 'endorsements': 3}
    ]
    career_history = [
        {
            'description': 'Built production ML systems using Python and PyTorch. Deployed vector search with Pinecone.',
            'company': 'Tech Corp',
            'duration_months': 48
        }
    ]
    
    score = skill_coherence_score(skills, career_history)
    assert score > 0.7, f"Expected >0.7 for high coherence, got {score}"


def test_skill_coherence_score_honeypot():
    """Test honeypot pattern (expert skills with 0 duration)"""
    skills = [
        {'name': 'Machine Learning', 'proficiency': 'expert', 'duration_months': 0, 'endorsements': 0},
        {'name': 'Deep Learning', 'proficiency': 'expert', 'duration_months': 0, 'endorsements': 0},
        {'name': 'NLP', 'proficiency': 'expert', 'duration_months': 0, 'endorsements': 0}
    ]
    career_history = [
        {'description': 'Managed marketing campaigns and customer relationships.', 'company': 'Services Co', 'duration_months': 48}
    ]
    
    score = skill_coherence_score(skills, career_history)
    assert score < 0.3, f"Expected <0.3 for honeypot pattern, got {score}"


def test_production_seniority_score_high():
    """Test high production seniority with vector DB and evaluation"""
    career_history = [
        {
            'description': 'Built production retrieval system using Pinecone and FAISS. Evaluated with NDCG and A/B testing.',
            'company': 'Product Co',
            'duration_months': 36
        }
    ]
    
    score = production_seniority_score(career_history)
    assert score > 0.7, f"Expected >0.7 for production evidence, got {score}"


def test_production_seniority_score_consulting():
    """Test penalty for pure consulting history"""
    career_history = [
        {
            'description': 'Data engineering work for enterprise clients.',
            'company': 'TCS',
            'duration_months': 48
        },
        {
            'description': 'Backend development.',
            'company': 'Infosys',
            'duration_months': 36
        }
    ]
    
    score = production_seniority_score(career_history)
    assert score < 0.5, f"Expected <0.5 for pure consulting, got {score}"


def test_location_fit_score_preferred():
    """Test preferred location returns 1.0"""
    profile = {'location': 'Hyderabad, Telangana', 'country': 'India'}
    redrob_signals = {'willing_to_relocate': False}
    
    score = location_fit_score(profile, redrob_signals)
    assert score == 1.0, f"Expected 1.0 for Hyderabad, got {score}"


def test_location_fit_score_india_other():
    """Test other Indian city returns 0.8"""
    profile = {'location': 'Bangalore, Karnataka', 'country': 'India'}
    redrob_signals = {'willing_to_relocate': False}
    
    score = location_fit_score(profile, redrob_signals)
    assert score == 0.8, f"Expected 0.8 for other Indian city, got {score}"


def test_location_fit_score_outside_india_relocate():
    """Test outside India but willing to relocate returns 0.6"""
    profile = {'location': 'London', 'country': 'UK'}
    redrob_signals = {'willing_to_relocate': True}
    
    score = location_fit_score(profile, redrob_signals)
    assert score == 0.6, f"Expected 0.6 for outside India with relocation, got {score}"


def test_disqualifier_penalty_consulting_only():
    """Test penalty for pure consulting history"""
    profile = {'current_title': 'Data Engineer'}
    career_history = [
        {'company': 'TCS', 'duration_months': 48},
        {'company': 'Infosys', 'duration_months': 36}
    ]
    
    penalty = disqualifier_penalty(profile, career_history)
    assert penalty < -0.2, f"Expected penalty < -0.2 for pure consulting, got {penalty}"


def test_disqualifier_penalty_title_chasing():
    """Test penalty for title-chasing pattern"""
    profile = {'current_title': 'Staff Engineer'}
    career_history = [
        {'company': 'Co A', 'duration_months': 12},
        {'company': 'Co B', 'duration_months': 14},
        {'company': 'Co C', 'duration_months': 16}
    ]
    
    penalty = disqualifier_penalty(profile, career_history)
    assert penalty < -0.1, f"Expected penalty for short tenures, got {penalty}"


def test_behavioral_multiplier_inactive():
    """Test multiplier for inactive candidate"""
    from datetime import datetime, timedelta
    
    # 6 months ago
    last_active = (datetime.now() - timedelta(days=200)).isoformat()
    redrob_signals = {
        'last_active_date': last_active,
        'recruiter_response_rate': 0.05,
        'open_to_work_flag': True,
        'interview_completion_rate': 0.8
    }
    
    multiplier = behavioral_multiplier(redrob_signals)
    assert multiplier < 0.6, f"Expected <0.6 for inactive candidate, got {multiplier}"


def test_behavioral_multiplier_active():
    """Test multiplier for active candidate"""
    from datetime import datetime, timedelta
    
    # Recently active
    last_active = (datetime.now() - timedelta(days=10)).isoformat()
    redrob_signals = {
        'last_active_date': last_active,
        'recruiter_response_rate': 0.9,
        'open_to_work_flag': True,
        'interview_completion_rate': 0.95
    }
    
    multiplier = behavioral_multiplier(redrob_signals)
    assert multiplier == 1.0, f"Expected 1.0 for active candidate, got {multiplier}"


def test_honeypot_flag_expert_zero_duration():
    """Test honeypot flag for expert skills with 0 duration"""
    profile = {'years_of_experience': 5}
    skills = [
        {'name': 'ML', 'proficiency': 'expert', 'duration_months': 0},
        {'name': 'DL', 'proficiency': 'expert', 'duration_months': 0},
        {'name': 'NLP', 'proficiency': 'expert', 'duration_months': 0}
    ]
    career_history = [{'duration_months': 60}]
    education = []
    
    flag = honeypot_flag(profile, skills, career_history, education)
    assert flag == True, f"Expected True for expert skills with 0 duration"


def test_honeypot_flag_experience_mismatch():
    """Test honeypot flag for experience mismatch"""
    profile = {'years_of_experience': 10}
    skills = []
    career_history = [{'duration_months': 36}]  # 3 years vs 10 claimed
    education = []
    
    flag = honeypot_flag(profile, skills, career_history, education)
    assert flag == True, f"Expected True for experience mismatch"


def test_honeypot_flag_invalid_education():
    """Test honeypot flag for invalid education dates"""
    profile = {'years_of_experience': 5}
    skills = []
    career_history = [{'duration_months': 60}]
    education = [{'start_year': 2020, 'end_year': 2015}]  # Invalid: start > end
    
    flag = honeypot_flag(profile, skills, career_history, education)
    assert flag == True, f"Expected True for invalid education dates"


def test_honeypot_penalty_multiplier():
    """Test honeypot penalty multiplier"""
    profile = {'years_of_experience': 5}
    skills = [
        {'name': 'ML', 'proficiency': 'expert', 'duration_months': 0},
        {'name': 'DL', 'proficiency': 'expert', 'duration_months': 0},
        {'name': 'NLP', 'proficiency': 'expert', 'duration_months': 0}
    ]
    career_history = [{'duration_months': 60}]
    education = []
    
    multiplier = honeypot_penalty_multiplier(profile, skills, career_history, education)
    assert multiplier == 0.05, f"Expected 0.05 for honeypot, got {multiplier}"


def test_reasoning_concrete_facts():
    """Test reasoning includes concrete facts (FIX 2 regression test)"""
    features = {
        'title_role_fit_score': 1.0,
        'experience_fit_score': 1.0,
        'skill_coherence_score': 0.8,
        'production_seniority_score': 0.7,
        'location_fit_score': 1.0,
        'disqualifier_penalty': 0.0,
        'behavioral_multiplier': 1.0,
        'honeypot_flag': False,
        'honeypot_penalty_multiplier': 1.0
    }
    score = 0.95
    
    candidate = {
        'profile': {
            'current_title': 'Senior ML Engineer',
            'current_company': 'Zomato',
            'years_of_experience': 7.2,
            'location': 'Bangalore',
            'country': 'India'
        },
        'skills': [
            {'name': 'PyTorch', 'proficiency': 'expert', 'duration_months': 48},
            {'name': 'Pinecone', 'proficiency': 'advanced', 'duration_months': 24}
        ],
        'career_history': [],
        'redrob_signals': {'willing_to_relocate': False}
    }
    
    reasoning = generate_reasoning(features, score, candidate)
    
    # Check for concrete facts
    assert '7.2' in reasoning or '7' in reasoning, f"Reasoning should contain years of experience: {reasoning}"
    assert 'Senior ML Engineer' in reasoning, f"Reasoning should contain title: {reasoning}"
    assert 'Zomato' in reasoning, f"Reasoning should contain company: {reasoning}"


def test_tiebreaker_unique_scores():
    """Test tie-breaker produces unique scores for perfect candidates (FIX 3 regression test)"""
    # Two candidates with identical perfect features
    features = {
        'title_role_fit_score': 1.0,
        'experience_fit_score': 1.0,
        'skill_coherence_score': 1.0,
        'production_seniority_score': 1.0,
        'location_fit_score': 1.0,
        'disqualifier_penalty': 0.0,
        'behavioral_multiplier': 1.0,
        'honeypot_penalty_multiplier': 1.0
    }
    
    score1 = compute_score(features, 'CAND_0018499')
    score2 = compute_score(features, 'CAND_0068811')
    
    # Scores should be different due to tie-breaker
    assert score1 != score2, f"Tie-breaker should produce unique scores: {score1} vs {score2}"
    assert abs(score1 - score2) < 0.01, f"Tie-breaker difference should be small: {abs(score1 - score2)}"


# ============ TESTS FOR EXPANDED FEATURES (Phase 11) ============

def test_experience_features():
    """Test experience feature extraction"""
    profile = {'years_of_experience': 7}
    career_history = [
        {'title': 'ML Engineer', 'company': 'Product Co', 'duration_months': 48},
        {'title': 'Data Scientist', 'company': 'Tech Corp', 'duration_months': 24}
    ]
    
    feats = experience_features(profile, career_history)
    
    assert feats['total_years'] == 7
    assert feats['relevant_years'] > 0
    assert 0 <= feats['seniority_level'] <= 1
    assert 0 <= feats['domain_experience'] <= 1


def test_skill_features():
    """Test skill feature extraction"""
    skills = [
        {'name': 'Python', 'proficiency': 'expert', 'duration_months': 48},
        {'name': 'PyTorch', 'proficiency': 'advanced', 'duration_months': 24},
        {'name': 'SQL', 'proficiency': 'intermediate', 'duration_months': 12}
    ]
    
    feats = skill_features(skills, [])
    
    assert feats['skill_count'] == 3
    assert feats['expert_skill_count'] == 1
    assert 0 <= feats['skill_diversity'] <= 1
    assert 0 <= feats['skill_recency'] <= 1


def test_career_features():
    """Test career feature extraction"""
    career_history = [
        {'title': 'Senior ML Engineer', 'company': 'Product Co', 'duration_months': 36},
        {'title': 'ML Engineer', 'company': 'Tech Corp', 'duration_months': 24}
    ]
    
    feats = career_features(career_history)
    
    assert feats['average_tenure'] > 0
    assert 0 <= feats['career_consistency'] <= 1
    assert 0 <= feats['employer_quality'] <= 1


def test_education_features():
    """Test education feature extraction"""
    education = [
        {'degree': 'B.Tech', 'field_of_study': 'Computer Science', 'institution': 'IIT Delhi', 'start_year': 2015, 'end_year': 2019}
    ]
    
    feats = education_features(education)
    
    assert feats['degree_relevance'] > 0
    assert feats['education_quality'] > 0
    assert 0 <= feats['academic_consistency'] <= 1


def test_behavioral_features():
    """Test behavioral feature extraction"""
    profile = {'current_title': 'ML Engineer', 'current_company': 'Tech Co', 'years_of_experience': 5, 'location': 'Bangalore'}
    redrob_signals = {
        'last_active_date': '2024-01-15T00:00:00Z',
        'recruiter_response_rate': 0.8,
        'open_to_work_flag': True,
        'interview_completion_rate': 0.9
    }
    
    feats = behavioral_features(profile, redrob_signals)
    
    assert 0 <= feats['profile_completeness'] <= 1
    assert 0 <= feats['engagement_score'] <= 1


def test_resume_quality_features():
    """Test resume quality feature extraction"""
    profile = {'current_title': 'ML Engineer', 'current_company': 'Tech Co', 'years_of_experience': 5}
    career_history = [
        {'description': 'Built production ML systems with PyTorch and deployed to AWS.' * 10}
    ]
    skills = [{'name': 'Python'}]
    
    feats = resume_quality_features(profile, career_history, skills)
    
    assert 0 <= feats['profile_completeness_score'] <= 1
    assert feats['project_richness'] > 0


def test_honeypot_flag_overlapping_jobs():
    """Test honeypot flag for overlapping jobs (Phase 5 enhancement)"""
    profile = {'years_of_experience': 5}
    skills = []
    career_history = [
        {'duration_months': 60},
        {'duration_months': 60}
    ]  # 120 months total vs 5 years claimed = impossible
    education = []
    
    flag = honeypot_flag(profile, skills, career_history, education)
    assert flag == True, "Should flag overlapping jobs"


def test_honeypot_flag_impossible_promotion():
    """Test honeypot flag for impossible promotion (Phase 5 enhancement)"""
    profile = {'years_of_experience': 5}
    skills = []
    career_history = [
        {'title': 'Junior Engineer', 'duration_months': 6},
        {'title': 'Senior Engineer', 'duration_months': 24}
    ]
    education = []
    
    flag = honeypot_flag(profile, skills, career_history, education)
    assert flag == True, "Should flag impossible promotion (junior to senior in 6 months)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
