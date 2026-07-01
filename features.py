from typing import Dict, Any, List
from datetime import datetime, timedelta
from semantic_features import SemanticFeatures
from utils import safe_get, safe_float, safe_int, safe_string
from disqualifiers import DisqualifierScorer


def extract_features(candidate: Dict[str, Any]) -> Dict[str, Any]:
    
    profile = safe_get(candidate, 'profile', {})
    career_history = safe_get(candidate, 'career_history', [])
    skills = safe_get(candidate, 'skills', [])
    education = safe_get(candidate, 'education', [])
    redrob_signals = safe_get(candidate, 'redrob_signals', {})
    certifications = safe_get(candidate, 'certifications', [])
    languages = safe_get(candidate, 'languages', [])
    
    
    semantic_features_dict = {}
    try:
        semantic = SemanticFeatures()
        semantic_features_dict = semantic.compute_all_semantic_features(candidate)
    except (ValueError, FileNotFoundError):
        
        semantic_features_dict = {
            'semantic_similarity': 0.0,
            'title_semantic_similarity': 0.0,
            'skill_jd_overlap': 0.0,
            'career_description_semantic_similarity': 0.0,
            'company_name_semantic_similarity': 0.0,
            'skill_name_semantic_similarity': 0.0,
            'education_field_semantic_similarity': 0.0,
        }
    
    
    exp_feats = experience_features(profile, career_history)
    skill_feats = skill_features(skills, career_history)
    career_feats = career_features(career_history)
    edu_feats = education_features(education)
    behav_feats = behavioral_features(profile, redrob_signals)
    redrob_ext_feats = redrob_extended_features(redrob_signals)
    cert_feats = certification_features(certifications)
    lang_feats = language_features(languages)
    company_feats = company_size_features(career_history)
    industry_feats = industry_transition_features(career_history)
    resume_feats = resume_quality_features(profile, career_history, skills)
    
    features = {
        
        'title_role_fit_score': title_role_fit_score(profile, career_history),
        'experience_fit_score': experience_fit_score(profile, career_history),
        'skill_coherence_score': skill_coherence_score(skills, career_history),
        'production_seniority_score': production_seniority_score(career_history),
        'location_fit_score': location_fit_score(profile, redrob_signals),
        'disqualifier_penalty': disqualifier_penalty(profile, career_history),
        'behavioral_multiplier': behavioral_multiplier(redrob_signals),
        'honeypot_flag': honeypot_flag(profile, skills, career_history, education),
        'honeypot_penalty_multiplier': honeypot_penalty_multiplier(profile, skills, career_history, education, redrob_signals),
        'honeypot_anomaly_score': honeypot_anomaly_score(profile, skills, career_history, education, redrob_signals),
        
        
        'semantic_similarity': semantic_features_dict['semantic_similarity'],
        'title_semantic_similarity': semantic_features_dict['title_semantic_similarity'],
        'skill_jd_overlap': semantic_features_dict['skill_jd_overlap'],
        'career_description_semantic_similarity': semantic_features_dict['career_description_semantic_similarity'],
        'company_name_semantic_similarity': semantic_features_dict['company_name_semantic_similarity'],
        'skill_name_semantic_similarity': semantic_features_dict['skill_name_semantic_similarity'],
        'education_field_semantic_similarity': semantic_features_dict['education_field_semantic_similarity'],
        
        
        'total_years': exp_feats['total_years'],
        'relevant_years': exp_feats['relevant_years'],
        'seniority_level': exp_feats['seniority_level'],
        'domain_experience': exp_feats['domain_experience'],
        'management_experience': exp_feats['management_experience'],
        
        #
        'skill_count': skill_feats['skill_count'],
        'expert_skill_count': skill_feats['expert_skill_count'],
        'skill_diversity': skill_feats['skill_diversity'],
        'skill_recency': skill_feats['skill_recency'],
        'rare_skill_score': skill_feats['rare_skill_score'],
        
       
        'promotion_frequency': career_feats['promotion_frequency'],
        'average_tenure': career_feats['average_tenure'],
        'career_growth_velocity': career_feats['career_growth_velocity'],
        'career_consistency': career_feats['career_consistency'],
        'employer_quality': career_feats['employer_quality'],
        'leadership_progression': career_feats['leadership_progression'],
        
        
        'degree_relevance': edu_feats['degree_relevance'],
        'education_quality': edu_feats['education_quality'],
        'certification_count': edu_feats['certification_count'],
        'academic_consistency': edu_feats['academic_consistency'],
        'education_tier_score': edu_feats['education_tier_score'],
        
        
        'profile_completeness': behav_feats['profile_completeness'],
        'profile_freshness': behav_feats['profile_freshness'],
        'recent_activity_score': behav_feats['recent_activity_score'],
        'career_momentum': behav_feats['career_momentum'],
        'engagement_score': behav_feats['engagement_score'],
        'employment_stability': behav_feats['employment_stability'],
        
        
        'redrob_profile_completeness_score': redrob_ext_feats['profile_completeness_score'],
        'profile_visibility_score': redrob_ext_feats['profile_visibility_score'],
        'application_activity_score': redrob_ext_feats['application_activity_score'],
        'response_speed_score': redrob_ext_feats['response_speed_score'],
        'skill_assessment_avg_score': redrob_ext_feats['skill_assessment_avg_score'],
        'social_proof_score': redrob_ext_feats['social_proof_score'],
        'availability_score': redrob_ext_feats['availability_score'],
        'salary_alignment_score': redrob_ext_feats['salary_alignment_score'],
        'work_mode_match_score': redrob_ext_feats['work_mode_match_score'],
        'github_activity_normalized': redrob_ext_feats['github_activity_normalized'],
        'recruiter_interest_score': redrob_ext_feats['recruiter_interest_score'],
        'offer_history_score': redrob_ext_feats['offer_history_score'],
        'verification_score': redrob_ext_feats['verification_score'],
        'platform_engagement_composite': redrob_ext_feats['platform_engagement_composite'],
        
        
        'profile_completeness_score': resume_feats['profile_completeness_score'],
        'missing_info_count': resume_feats['missing_info_count'],
        'project_richness': resume_feats['project_richness'],
        
       
        'certification_count': cert_feats['certification_count'],
        'certification_relevance_score': cert_feats['certification_relevance_score'],
        'certification_recency_score': cert_feats['certification_recency_score'],
        'certification_quality_score': cert_feats['certification_quality_score'],
        
        
        'language_count': lang_feats['language_count'],
        'english_proficiency_score': lang_feats['english_proficiency_score'],
        'native_language_count': lang_feats['native_language_count'],
        'professional_language_count': lang_feats['professional_language_count'],
        
        
        'company_size_progression': company_feats['company_size_progression'],
        'current_company_size_score': company_feats['current_company_size_score'],
        'avg_company_size_score': company_feats['avg_company_size_score'],
        
       
        'industry_consistency': industry_feats['industry_consistency'],
        'industry_diversity': industry_feats['industry_diversity'],
        'current_industry_relevance': industry_feats['current_industry_relevance'],
    }
    
    return features


def title_role_fit_score(profile: Dict[str, Any], career_history: List[Dict[str, Any]]) -> float:
    
    current_title = profile.get('current_title', '').lower()
    
   
    recent_titles = [role['title'].lower() for role in career_history[:3]]
    all_titles = [current_title] + recent_titles
    
    
    tier_a_keywords = [
        'ml engineer', 'machine learning engineer', 'ai engineer',
        'data scientist', 'applied scientist',
        'search engineer', 'ranking engineer', 'retrieval engineer',
        'recommendation engineer', 'recommendation systems engineer', 
        'nlp engineer', 'senior nlp engineer',  
        'ai research engineer',  
        'senior software engineer (ml)',  
        'ml ops', 'mlops'
    ]
    
    
    
    tier_b_keywords = [
        'data engineer', 'backend engineer', 'software engineer',
        'analytics engineer', 'research engineer',
        'full stack engineer', 'full-stack engineer',
        'computer vision engineer'  
    ]
    
    
    tier_c_keywords = [
        'frontend engineer', 'mobile developer', 'android developer',
        'ios developer', 'devops engineer', 'qa engineer',
        'business analyst', 'product manager', 'project manager'
    ]
    
    
    tier_d_keywords = [
        'marketing manager', 'hr manager', 'human resources',
        'accountant', 'operations manager', 'customer support',
        'civil engineer', 'mechanical engineer'
    ]
    
    
    best_score = 0.0
    
    for title in all_titles:
        
        if any(keyword in title for keyword in tier_d_keywords):
            
            if title == current_title:
                return 0.0
            best_score = max(best_score, 0.0)
        
        
        elif any(keyword in title for keyword in tier_a_keywords):
            return 1.0
        
        
        elif any(keyword in title for keyword in tier_b_keywords):
            best_score = max(best_score, 0.6)
        
        
        elif any(keyword in title for keyword in tier_c_keywords):
            best_score = max(best_score, 0.3)
    
    return best_score


def experience_fit_score(profile: Dict[str, Any], career_history: List[Dict[str, Any]]) -> float:
  
    years_exp = profile.get('years_of_experience', 0)
    
    if 5 <= years_exp <= 9:
        return 1.0
    elif 4 <= years_exp < 5 or 9 < years_exp <= 11:
        return 0.8
    elif 3 <= years_exp < 4 or 11 < years_exp <= 13:
        return 0.5
    elif 1 <= years_exp < 3 or 13 < years_exp <= 15:
        return 0.2
    else:
        return 0.0


def skill_coherence_score(skills: List[Dict[str, Any]], career_history: List[Dict[str, Any]]) -> float:
    
    if not skills:
        return 0.0
    
    
    career_text = ' '.join([role.get('description', '').lower() for role in career_history])
    
    
    relevant_skills = [
        'python', 'machine learning', 'ml', 'deep learning', 'pytorch', 'tensorflow',
        'scikit-learn', 'numpy', 'pandas',
        'embeddings', 'vector database', 'pinecone', 'weaviate', 'qdrant', 'milvus',
        'elasticsearch', 'opensearch', 'faiss',
        'retrieval', 'ranking', 'recommendation', 'search',
        'llm', 'langchain', 'rag', 'fine-tuning', 'nlp',
        'airflow', 'spark', 'kafka', 'sql'
    ]
    
    proficiency_weights = {
        'expert': 1.0,
        'advanced': 0.8,
        'intermediate': 0.5,
        'beginner': 0.2
    }
    
    total_weight = 0.0
    total_duration = 0
    skills_in_career = 0
    
    for skill in skills:
        skill_name = skill.get('name', '').lower()
        proficiency = skill.get('proficiency', 'beginner')
        duration = skill.get('duration_months', 0)
        
        
        is_relevant = any(rs in skill_name for rs in relevant_skills)
        
        if is_relevant:
            prof_weight = proficiency_weights.get(proficiency, 0.2)
            
            
            if duration == 0 and proficiency == 'expert':
                prof_weight *= 0.1
            
            elif duration == 0:
                prof_weight *= 0.3
            
            
            if skill_name in career_text:
                prof_weight *= 1.5  
                skills_in_career += 1
            
            
            duration_weight = min(duration / 60.0, 1.0) if duration > 0 else 0.1
            
            total_weight += prof_weight * duration_weight
            total_duration += duration
    
    
    relevant_count = sum(1 for s in skills if any(rs in s.get('name', '').lower() for rs in relevant_skills))
    
    if relevant_count == 0:
        return 0.0
    
    base_score = total_weight / relevant_count
    
    
    career_ratio = skills_in_career / relevant_count if relevant_count > 0 else 0
    base_score *= (1.0 + 0.5 * career_ratio)
    
    
    return min(max(base_score, 0.0), 1.0)


def production_seniority_score(career_history: List[Dict[str, Any]]) -> float:
    
    consulting_companies = [
        'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'
    ]
    
    vector_db_keywords = [
        'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'faiss'
    ]
    
    evaluation_keywords = [
        'ndcg', 'mrr', 'map', 'a/b test', 'ab test', 'evaluation', 'offline', 'online'
    ]
    
    production_keywords = [
        'production', 'deployed', 'shipped', 'launched', 'live', 'real users',
        'scale', 'million', 'billion', 'high-traffic', 'latency'
    ]
    
    has_product_company = False
    has_consulting_only = True
    has_vector_db = False
    has_evaluation = False
    has_production = False
    
    for role in career_history:
        company = role.get('company', '').lower()
        description = role.get('description', '').lower()
        
        
        if not any(consult in company for consult in consulting_companies):
            has_product_company = True
            has_consulting_only = False
        
        
        if any(vdb in description for vdb in vector_db_keywords):
            has_vector_db = True
        
        
        if any(eval_kw in description for eval_kw in evaluation_keywords):
            has_evaluation = True
        
        
        if any(prod_kw in description for prod_kw in production_keywords):
            has_production = True
    
   
    tech_score = 0.0
    if has_vector_db:
        tech_score += 0.4
    if has_evaluation:
        tech_score += 0.3
    if has_production:
        tech_score += 0.3
    
    
    if has_product_company:
        tech_score *= 1.2  
    elif has_consulting_only:
        tech_score *= 0.5 
    
    
    return min(max(tech_score, 0.0), 1.0)


def location_fit_score(profile: Dict[str, Any], redrob_signals: Dict[str, Any]) -> float:
    
    location = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
    willing_to_relocate = redrob_signals.get('willing_to_relocate', False)
    
    preferred_cities = ['hyderabad', 'pune', 'mumbai', 'delhi', 'noida', 'gurgaon']
    
   
    if any(city in location for city in preferred_cities):
        return 1.0
    
    
    if country == 'india':
        return 0.8
    
    
    if willing_to_relocate:
        return 0.6
    
  
    return 0.0


def disqualifier_penalty(profile: Dict[str, Any], career_history: List[Dict[str, Any]]) -> float:
    
    scorer = DisqualifierScorer()
    
    
    candidate = {
        'profile': profile,
        'career_history': career_history,
        'redrob_signals': {},  
    }
    
    result = scorer.compute_disqualifier_score(candidate)
    
    
    disqualifier_score = result['disqualifier_score']
    
    return -disqualifier_score


def behavioral_multiplier(redrob_signals: Dict[str, Any]) -> float:
   
    multiplier = 1.0
    
   
    last_active = redrob_signals.get('last_active_date', '')
    if last_active:
        try:
            last_active_date = datetime.fromisoformat(last_active.replace('Z', '+00:00'))
            if last_active_date.tzinfo:
                last_active_date = last_active_date.replace(tzinfo=None)
            days_inactive = (datetime.now() - last_active_date).days
            
            if days_inactive > 180:  
                multiplier *= 0.5
            elif days_inactive > 90: 
                multiplier *= 0.7
        except:
            pass
    
   
    response_rate = redrob_signals.get('recruiter_response_rate', 1.0)
    if response_rate < 0.1: 
        multiplier *= 0.5
    elif response_rate < 0.3: 
        multiplier *= 0.7
    
    
    if not redrob_signals.get('open_to_work_flag', True):
        multiplier *= 0.8
    
    
    interview_rate = redrob_signals.get('interview_completion_rate', 1.0)
    if interview_rate < 0.5: 
        multiplier *= 0.7
    
    
    return max(min(multiplier, 1.0), 0.3)


def honeypot_flag(profile: Dict[str, Any], skills: List[Dict[str, Any]], 
                  career_history: List[Dict[str, Any]], education: List[Dict[str, Any]]) -> bool:
    
    
    expert_zero_duration = sum(
        1 for s in skills
        if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0
    )
    if expert_zero_duration >= 3:
        return True
    
   
    total_career_months = sum(role.get('duration_months', 0) for role in career_history)
    career_years = total_career_months / 12.0
    profile_years = profile.get('years_of_experience', 0)
    
    if abs(career_years - profile_years) > 3:
        return True
    
    
    for edu in education:
        start_year = edu.get('start_year', 0)
        end_year = edu.get('end_year', 0)
        
        if start_year > end_year:  
            return True
        if end_year - start_year > 12:  
            return True
    
    
    if len(career_history) >= 2:
       
        total_duration = sum(role.get('duration_months', 0) for role in career_history)
        if total_duration > (profile_years + 5) * 12: 
            return True
    
    
    if len(career_history) >= 3:
        senior_keywords = ['senior', 'lead', 'principal', 'staff', 'architect', 'director']
        junior_keywords = ['junior', 'intern', 'associate']
        
        
        for i in range(len(career_history) - 1):
            curr_title = career_history[i].get('title', '').lower()
            next_title = career_history[i+1].get('title', '').lower()
            curr_duration = career_history[i].get('duration_months', 0)
            
            curr_is_junior = any(kw in curr_title for kw in junior_keywords)
            next_is_senior = any(kw in next_title for kw in senior_keywords)
            
            if curr_is_junior and next_is_senior and curr_duration < 12:
                return True
    
    
    if profile_years > 30:  
    
    
     if len(skills) > 50:  
        
        expert_advanced_count = sum(
            1 for s in skills 
            if s.get('proficiency') in ['expert', 'advanced']
        )
        if expert_advanced_count / len(skills) > 0.8:
            return True
    
   
    current_title = profile.get('current_title', '').lower()
    if career_history:
        first_role_title = career_history[0].get('title', '').lower()
        
        current_words = set(current_title.split())
        role_words = set(first_role_title.split())
        if current_words and role_words and len(current_words & role_words) == 0:
            
            pass 
    
    return False


def honeypot_anomaly_score(profile: Dict[str, Any], skills: List[Dict[str, Any]], 
                           career_history: List[Dict[str, Any]], education: List[Dict[str, Any]],
                           redrob_signals: Dict[str, Any]) -> float:
    
    import numpy as np
    
    features = []
    
   
    skill_count = len(skills)
    features.append(min(skill_count / 100, 1.0))  
    
    
    if skills:
        expert_count = sum(1 for s in skills if s.get('proficiency') == 'expert')
        expert_ratio = expert_count / len(skills)
        features.append(expert_ratio)
    else:
        features.append(0.0)
    
   
    total_career_months = sum(role.get('duration_months', 0) for role in career_history)
    career_years = total_career_months / 12.0
    profile_years = profile.get('years_of_experience', 0)
    exp_mismatch = abs(career_years - profile_years) / (profile_years + 1)
    features.append(min(exp_mismatch, 1.0))
    
    
    if career_history:
        durations = [role.get('duration_months', 0) for role in career_history]
        avg_duration = np.mean(durations) if durations else 0
        std_duration = np.std(durations) if len(durations) > 1 else 0
        
        duration_inconsistency = std_duration / (avg_duration + 1)
        features.append(min(duration_inconsistency, 1.0))
    else:
        features.append(0.0)
    
    
    profile_completeness = redrob_signals.get('profile_completeness_score', 50) / 100.0
    
    if profile_completeness < 0.3:
        completeness_anomaly = 1.0 - profile_completeness / 0.3
    elif profile_completeness > 0.95:
        completeness_anomaly = (profile_completeness - 0.95) / 0.05
    else:
        completeness_anomaly = 0.0
    features.append(min(completeness_anomaly, 1.0))
    

    profile_views = redrob_signals.get('profile_views_received_30d', 0)
    applications = redrob_signals.get('applications_submitted_30d', 0)
    if profile_views > 50 and applications == 0:
        behavioral_anomaly = 1.0 
    else:
        behavioral_anomaly = 0.0
    features.append(behavioral_anomaly)
    
    
    if len(education) >= 2:
        edu_durations = []
        for edu in education:
            start = edu.get('start_year', 0)
            end = edu.get('end_year', 0)
            if end > start:
                edu_durations.append(end - start)
        if edu_durations:
            avg_edu_duration = np.mean(edu_durations)
            
            edu_anomaly = 1.0 - min(avg_edu_duration / 3, 1.0)  # Normalize by 3 years
        else:
            edu_anomaly = 0.0
    else:
        edu_anomaly = 0.0
    features.append(edu_anomaly)
    
    
    weights = [0.2, 0.15, 0.2, 0.15, 0.1, 0.1, 0.1]
    anomaly_score = sum(w * f for w, f in zip(weights, features))
    
    return min(anomaly_score, 1.0)


def honeypot_penalty_multiplier(profile: Dict[str, Any], skills: List[Dict[str, Any]],
                                  career_history: List[Dict[str, Any]], 
                                  education: List[Dict[str, Any]],
                                  redrob_signals: Dict[str, Any] = None) -> float:
    
    if redrob_signals is None:
        redrob_signals = {}
    
    
    is_honeypot = honeypot_flag(profile, skills, career_history, education)
    
    if is_honeypot:
        return 0.05  
    
   
    anomaly_score = honeypot_anomaly_score(profile, skills, career_history, education, redrob_signals)
    
    
    if anomaly_score < 0.3:
        return 1.0
    elif anomaly_score < 0.5:
        return 0.8
    elif anomaly_score < 0.7:
        return 0.5
    else:
        return 0.2



def experience_features(profile: Dict[str, Any], career_history: List[Dict[str, Any]]) -> Dict[str, float]:
    
    years_exp = profile.get('years_of_experience', 0)
    
    
    relevant_years = 0
    for role in career_history:
        title = role.get('title', '').lower()
        if any(kw in title for kw in ['ml', 'machine learning', 'ai', 'data scientist', 'nlp', 'deep learning']):
            relevant_years += role.get('duration_months', 0) / 12
    
    
    current_title = profile.get('current_title', '').lower()
    senior_keywords = ['senior', 'lead', 'principal', 'staff', 'architect', 'head of', 'director']
    seniority_level = 1.0 if any(kw in current_title for kw in senior_keywords) else 0.5
    if 'junior' in current_title or 'intern' in current_title:
        seniority_level = 0.2
    
   
    consulting_companies = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini']
    product_months = 0
    for role in career_history:
        company = role.get('company', '').lower()
        if not any(consult in company for consult in consulting_companies):
            product_months += role.get('duration_months', 0)
    domain_experience = min(product_months / (years_exp * 12 + 1), 1.0)
    
    
    management_keywords = ['manager', 'lead', 'director', 'head', 'principal', 'architect']
    management_months = 0
    for role in career_history:
        title = role.get('title', '').lower()
        if any(kw in title for kw in management_keywords):
            management_months += role.get('duration_months', 0)
    management_experience = min(management_months / (years_exp * 12 + 1), 1.0)
    
    return {
        'total_years': years_exp,
        'relevant_years': relevant_years,
        'seniority_level': seniority_level,
        'domain_experience': domain_experience,
        'management_experience': management_experience,
    }


def skill_features(skills: List[Dict[str, Any]], career_history: List[Dict[str, Any]]) -> Dict[str, float]:
    
    if not skills:
        return {
            'skill_count': 0,
            'expert_skill_count': 0,
            'skill_diversity': 0.0,
            'skill_recency': 0.0,
            'rare_skill_score': 0.0,
        }
    
    skill_count = len(skills)
    expert_count = sum(1 for s in skills if s.get('proficiency') == 'expert')
    
    
    categories = set()
    for skill in skills:
        name = skill.get('name', '').lower()
        if any(kw in name for kw in ['python', 'java', 'c++', 'javascript']):
            categories.add('programming')
        elif any(kw in name for kw in ['ml', 'machine learning', 'deep learning', 'ai']):
            categories.add('ml')
        elif any(kw in name for kw in ['nlp', 'text', 'language']):
            categories.add('nlp')
        elif any(kw in name for kw in ['database', 'sql', 'nosql']):
            categories.add('database')
        elif any(kw in name for kw in ['cloud', 'aws', 'gcp', 'azure']):
            categories.add('cloud')
    skill_diversity = len(categories) / 5.0 
    
   
    total_duration = sum(s.get('duration_months', 0) for s in skills)
    skill_recency = min(total_duration / (skill_count * 60), 1.0)  
    
    
    rare_skills = ['rag', 'fine-tuning', 'loRA', 'qlora', 'peft', 'vector database', 'pinecone', 'weaviate', 'qdrant', 'milvus']
    rare_count = sum(1 for s in skills if any(rare in s.get('name', '').lower() for rare in rare_skills))
    rare_skill_score = rare_count / len(skills) if skills else 0.0
    
    return {
        'skill_count': skill_count,
        'expert_skill_count': expert_count,
        'skill_diversity': skill_diversity,
        'skill_recency': skill_recency,
        'rare_skill_score': rare_skill_score,
    }


def career_features(career_history: List[Dict[str, Any]]) -> Dict[str, float]:
    
    if not career_history:
        return {
            'promotion_frequency': 0.0,
            'average_tenure': 0.0,
            'career_growth_velocity': 0.0,
            'career_consistency': 0.0,
            'employer_quality': 0.0,
            'leadership_progression': 0.0,
        }
    
    
    tenures = [role.get('duration_months', 0) for role in career_history]
    average_tenure = sum(tenures) / len(tenures) if tenures else 0
    
    
    seniority_keywords = ['senior', 'lead', 'principal', 'staff', 'architect', 'head of', 'director']
    promotions = 0
    for i in range(1, len(career_history)):
        prev_title = career_history[i-1].get('title', '').lower()
        curr_title = career_history[i].get('title', '').lower()
        prev_senior = any(kw in prev_title for kw in seniority_keywords)
        curr_senior = any(kw in curr_title for kw in seniority_keywords)
        if not prev_senior and curr_senior:
            promotions += 1
    promotion_frequency = promotions / len(career_history) if career_history else 0
    
    
    title_lengths = [len(role.get('title', '')) for role in career_history]
    if len(title_lengths) > 1:
        career_growth_velocity = (title_lengths[0] - title_lengths[-1]) / len(title_lengths)
    else:
        career_growth_velocity = 0.0
    career_growth_velocity = max(min(career_growth_velocity, 1.0), 0.0)
    
    
    domains = set()
    for role in career_history:
        title = role.get('title', '').lower()
        if any(kw in title for kw in ['ml', 'ai', 'data', 'machine learning']):
            domains.add('ml')
        elif any(kw in title for kw in ['software', 'engineer', 'developer']):
            domains.add('software')
        elif any(kw in title for kw in ['manager', 'lead', 'director']):
            domains.add('management')
    career_consistency = 1.0 if len(domains) <= 2 else 0.5
    
    
    consulting_companies = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini']
    product_count = 0
    for role in career_history:
        company = role.get('company', '').lower()
        if not any(consult in company for consult in consulting_companies):
            product_count += 1
    employer_quality = product_count / len(career_history) if career_history else 0
    
    
    leadership_count = sum(1 for role in career_history if any(kw in role.get('title', '').lower() for kw in seniority_keywords))
    leadership_progression = leadership_count / len(career_history) if career_history else 0
    
    return {
        'promotion_frequency': promotion_frequency,
        'average_tenure': average_tenure,
        'career_growth_velocity': career_growth_velocity,
        'career_consistency': career_consistency,
        'employer_quality': employer_quality,
        'leadership_progression': leadership_progression,
    }


def education_features(education: List[Dict[str, Any]]) -> Dict[str, float]:
   
    if not education:
        return {
            'degree_relevance': 0.0,
            'education_quality': 0.0,
            'certification_count': 0,
            'academic_consistency': 0.0,
            'education_tier_score': 0.0,
        }
    
    
    relevant_degrees = ['computer science', 'machine learning', 'artificial intelligence', 'data science', 'mathematics', 'statistics']
    relevance_count = 0
    for edu in education:
        field = edu.get('field_of_study', '').lower()
        if any(relevant in field for relevant in relevant_degrees):
            relevance_count += 1
    degree_relevance = relevance_count / len(education) if education else 0
    
    
    tier_scores = {'tier_1': 1.0, 'tier_2': 0.75, 'tier_3': 0.5, 'tier_4': 0.25, 'unknown': 0.5}
    tier_values = []
    for edu in education:
        tier = edu.get('tier', 'unknown')
        tier_values.append(tier_scores.get(tier, 0.5))
    education_tier_score = sum(tier_values) / len(tier_values) if tier_values else 0.0
    
    
    tier1_keywords = ['iit', 'iim', 'bits', 'stanford', 'mit', 'berkeley', 'carnegie']
    quality_count = 0
    for edu in education:
        institution = edu.get('institution', '').lower()
        if any(tier1 in institution for tier1 in tier1_keywords):
            quality_count += 1
    education_quality = quality_count / len(education) if education else 0
    
    
    cert_count = 0
    for edu in education:
        degree = edu.get('degree', '').lower()
        if 'certification' in degree or 'certificate' in degree:
            cert_count += 1
    
    
    consistent_count = 0
    for edu in education:
        start = edu.get('start_year', 0)
        end = edu.get('end_year', 0)
        if start > 0 and end > 0 and start <= end and (end - start) <= 6:
            consistent_count += 1
    academic_consistency = consistent_count / len(education) if education else 0
    
    return {
        'degree_relevance': degree_relevance,
        'education_quality': education_quality,
        'certification_count': cert_count,
        'academic_consistency': academic_consistency,
        'education_tier_score': education_tier_score,
    }


def behavioral_features(profile: Dict[str, Any], redrob_signals: Dict[str, Any]) -> Dict[str, float]:
    
    completeness_fields = [
        profile.get('current_title'),
        profile.get('current_company'),
        profile.get('years_of_experience'),
        profile.get('location'),
    ]
    filled_fields = sum(1 for f in completeness_fields if f)
    profile_completeness = filled_fields / len(completeness_fields)
    
    
    last_active = redrob_signals.get('last_active_date', '')
    if last_active:
        try:
            from datetime import datetime
            last_active_date = datetime.fromisoformat(last_active.replace('Z', '+00:00'))
            if last_active_date.tzinfo:
                last_active_date = last_active_date.replace(tzinfo=None)
            days_inactive = (datetime.now() - last_active_date).days
            profile_freshness = max(0, 1 - days_inactive / 365)  # Decay over 1 year
        except:
            profile_freshness = 0.5
    else:
        profile_freshness = 0.0
    
    
    response_rate = redrob_signals.get('recruiter_response_rate', 0)
    open_to_work = redrob_signals.get('open_to_work_flag', False)
    interview_rate = redrob_signals.get('interview_completion_rate', 0)
    recent_activity_score = (response_rate * 0.4 + 
                            (1.0 if open_to_work else 0.5) * 0.3 + 
                            interview_rate * 0.3)
    
    
    career_momentum = 0.5  
    
    
    engagement_score = (profile_freshness * 0.4 + recent_activity_score * 0.6)
    
    
    employment_stability = 0.5 
    
    return {
        'profile_completeness': profile_completeness,
        'profile_freshness': profile_freshness,
        'recent_activity_score': recent_activity_score,
        'career_momentum': career_momentum,
        'engagement_score': engagement_score,
        'employment_stability': employment_stability,
    }


def redrob_extended_features(redrob_signals: Dict[str, Any]) -> Dict[str, float]:
    
    
    profile_completeness_score = redrob_signals.get('profile_completeness_score', 0) / 100.0
    
   
    views_30d = redrob_signals.get('profile_views_received_30d', 0)
    search_appearances = redrob_signals.get('search_appearance_30d', 0)
    profile_visibility_score = min((views_30d + search_appearances) / 100, 1.0)
    
    
    applications_30d = redrob_signals.get('applications_submitted_30d', 0)
    application_activity_score = min(applications_30d / 20, 1.0)
    
    
    avg_response_hours = redrob_signals.get('avg_response_time_hours', 24)
    response_speed_score = max(0, 1 - avg_response_hours / 72)  
    
   
    skill_assessments = redrob_signals.get('skill_assessment_scores', {})
    if skill_assessments:
        skill_assessment_avg_score = sum(skill_assessments.values()) / len(skill_assessments) / 100.0
    else:
        skill_assessment_avg_score = 0.0
    
    
    connection_count = redrob_signals.get('connection_count', 0)
    endorsements = redrob_signals.get('endorsements_received', 0)
    social_proof_score = min((connection_count + endorsements) / 500, 1.0)
    
   
    notice_period = redrob_signals.get('notice_period_days', 90)
    availability_score = max(0, 1 - notice_period / 180)  
    
    
    salary_range = redrob_signals.get('expected_salary_range_inr_lpa', {})
    salary_min = salary_range.get('min', 0)
    salary_max = salary_range.get('max', 0)
    
    if salary_min > 0 and salary_max > 0:
        overlap = min(salary_max, 50) - max(salary_min, 20)
        salary_alignment_score = max(0, overlap / 30) 
    else:
        salary_alignment_score = 0.5  
    
   
    preferred_mode = redrob_signals.get('preferred_work_mode', 'flexible')
    if preferred_mode in ['hybrid', 'onsite']:
        work_mode_match_score = 1.0
    elif preferred_mode == 'flexible':
        work_mode_match_score = 0.8
    else:  
        work_mode_match_score = 0.6
    
    
    github_score = redrob_signals.get('github_activity_score', -1)
    if github_score == -1:
        github_activity_normalized = 0.0  
    else:
        github_activity_normalized = github_score / 100.0
    
    
    saved_by_recruiters = redrob_signals.get('saved_by_recruiters_30d', 0)
    recruiter_interest_score = min(saved_by_recruiters / 10, 1.0)
    
    
    offer_acceptance = redrob_signals.get('offer_acceptance_rate', -1)
    if offer_acceptance == -1:
        offer_history_score = 0.5 
    else:
        offer_history_score = offer_acceptance  
    
    
    verified_email = redrob_signals.get('verified_email', False)
    verified_phone = redrob_signals.get('verified_phone', False)
    linkedin_connected = redrob_signals.get('linkedin_connected', False)
    verification_score = (verified_email + verified_phone + linkedin_connected) / 3.0
    
   
    platform_engagement_composite = (
        profile_visibility_score * 0.2 +
        application_activity_score * 0.15 +
        response_speed_score * 0.15 +
        social_proof_score * 0.15 +
        recruiter_interest_score * 0.15 +
        verification_score * 0.1 +
        github_activity_normalized * 0.1
    )
    
    return {
        'profile_completeness_score': profile_completeness_score,
        'profile_visibility_score': profile_visibility_score,
        'application_activity_score': application_activity_score,
        'response_speed_score': response_speed_score,
        'skill_assessment_avg_score': skill_assessment_avg_score,
        'social_proof_score': social_proof_score,
        'availability_score': availability_score,
        'salary_alignment_score': salary_alignment_score,
        'work_mode_match_score': work_mode_match_score,
        'github_activity_normalized': github_activity_normalized,
        'recruiter_interest_score': recruiter_interest_score,
        'offer_history_score': offer_history_score,
        'verification_score': verification_score,
        'platform_engagement_composite': platform_engagement_composite,
    }


def resume_quality_features(profile: Dict[str, Any], career_history: List[Dict[str, Any]], skills: List[Dict[str, Any]]) -> Dict[str, float]:
    
    missing_count = 0
    if not profile.get('current_title'):
        missing_count += 1
    if not profile.get('current_company'):
        missing_count += 1
    if not profile.get('years_of_experience'):
        missing_count += 1
    if not career_history:
        missing_count += 1
    if not skills:
        missing_count += 1
    
    
    total_desc_length = sum(len(role.get('description', '')) for role in career_history)
    project_richness = min(total_desc_length / 1000, 1.0)  # Normalize by 1000 chars
    
    return {
        'profile_completeness_score': 1.0 - (missing_count / 5),
        'missing_info_count': missing_count,
        'project_richness': project_richness,
    }


def certification_features(certifications: List[Dict[str, Any]]) -> Dict[str, float]:
    
    if not certifications:
        return {
            'certification_count': 0,
            'certification_relevance_score': 0.0,
            'certification_recency_score': 0.0,
            'certification_quality_score': 0.0,
        }
    
    certification_count = len(certifications)
    
    
    relevant_keywords = ['aws', 'gcp', 'azure', 'tensorflow', 'pytorch', 'kubernetes', 'docker', 'ml', 'ai', 'data science', 'machine learning']
    relevant_count = 0
    for cert in certifications:
        name = cert.get('name', '').lower()
        issuer = cert.get('issuer', '').lower()
        if any(kw in name or kw in issuer for kw in relevant_keywords):
            relevant_count += 1
    certification_relevance_score = relevant_count / certification_count if certification_count > 0 else 0.0
    
   
    current_year = 2026
    years_since = []
    for cert in certifications:
        year = cert.get('year', 0)
        if year > 0:
            years_since.append(current_year - year)
    if years_since:
        avg_years = sum(years_since) / len(years_since)
        certification_recency_score = max(0, 1 - avg_years / 5)  # Decay over 5 years
    else:
        certification_recency_score = 0.0
    
    
    high_quality_issuers = ['google', 'amazon', 'microsoft', 'ibm', 'meta', 'nvidia', 'coursera', 'udacity', 'stanford', 'mit']
    quality_count = 0
    for cert in certifications:
        issuer = cert.get('issuer', '').lower()
        if any(hq in issuer for hq in high_quality_issuers):
            quality_count += 1
    certification_quality_score = quality_count / certification_count if certification_count > 0 else 0.0
    
    return {
        'certification_count': certification_count,
        'certification_relevance_score': certification_relevance_score,
        'certification_recency_score': certification_recency_score,
        'certification_quality_score': certification_quality_score,
    }


def language_features(languages: List[Dict[str, Any]]) -> Dict[str, float]:
    
    if not languages:
        return {
            'language_count': 0,
            'english_proficiency_score': 0.0,
            'native_language_count': 0,
            'professional_language_count': 0,
        }
    
    language_count = len(languages)
    
    
    english_proficiency = 0.0
    for lang in languages:
        if lang.get('language', '').lower() == 'english':
            prof = lang.get('proficiency', 'basic')
            if prof == 'native':
                english_proficiency = 1.0
            elif prof == 'professional':
                english_proficiency = 0.8
            elif prof == 'conversational':
                english_proficiency = 0.5
            else:  # basic
                english_proficiency = 0.3
            break
    
    
    native_count = sum(1 for lang in languages if lang.get('proficiency') == 'native')
    professional_count = sum(1 for lang in languages if lang.get('proficiency') in ['professional', 'native'])
    
    return {
        'language_count': language_count,
        'english_proficiency_score': english_proficiency,
        'native_language_count': native_count,
        'professional_language_count': professional_count,
    }


def company_size_features(career_history: List[Dict[str, Any]]) -> Dict[str, float]:
    
    if not career_history:
        return {
            'company_size_progression': 0.0,
            'current_company_size_score': 0.0,
            'avg_company_size_score': 0.0,
        }
    
    
    size_scores = {
        '1-10': 0.1,
        '11-50': 0.2,
        '51-200': 0.4,
        '201-500': 0.6,
        '501-1000': 0.7,
        '1001-5000': 0.8,
        '5001-10000': 0.9,
        '10001+': 1.0
    }
    
    
    sizes = []
    for role in career_history:
        size = role.get('company_size', 'unknown')
        sizes.append(size_scores.get(size, 0.5))
    
    
    current_company_size_score = sizes[0] if sizes else 0.0
    
    
    avg_company_size_score = sum(sizes) / len(sizes) if sizes else 0.0
    
    
    if len(sizes) >= 2:
        
        mid = len(sizes) // 2
        early_avg = sum(sizes[:mid]) / mid if mid > 0 else 0.0
        late_avg = sum(sizes[mid:]) / (len(sizes) - mid) if len(sizes) > mid else 0.0
        company_size_progression = late_avg - early_avg + 0.5  # Center around 0.5
        company_size_progression = max(0, min(company_size_progression, 1.0))
    else:
        company_size_progression = 0.5
    
    return {
        'company_size_progression': company_size_progression,
        'current_company_size_score': current_company_size_score,
        'avg_company_size_score': avg_company_size_score,
    }


def industry_transition_features(career_history: List[Dict[str, Any]]) -> Dict[str, float]:
    
    if not career_history:
        return {
            'industry_consistency': 0.0,
            'industry_diversity': 0.0,
            'current_industry_relevance': 0.0,
        }
    
   
    industries = [role.get('industry', '').lower() for role in career_history]
    
    
    unique_industries = set(industries)
    industry_diversity = len(unique_industries) / len(industries) if industries else 0.0
    
    
    if len(industries) >= 2:
        same_industry_count = sum(1 for i in range(len(industries)-1) if industries[i] == industries[i+1])
        industry_consistency = same_industry_count / (len(industries) - 1)
    else:
        industry_consistency = 1.0
    
    
    relevant_industries = ['technology', 'software', 'artificial intelligence', 'machine learning', 'data science', 'computer software', 'internet']
    current_industry = industries[0] if industries else ''
    current_industry_relevance = 1.0 if any(rel in current_industry for rel in relevant_industries) else 0.3
    
    return {
        'industry_consistency': industry_consistency,
        'industry_diversity': industry_diversity,
        'current_industry_relevance': current_industry_relevance,
    }
