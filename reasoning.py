
from typing import Dict, Any


def generate_reasoning(features: Dict[str, Any], score: float, candidate: Dict[str, Any] = None, tier: str = None) -> str:
    
    if candidate:
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

        best_skill = None
        for skill in skills:
            if skill.get('proficiency') in ['expert', 'advanced'] and skill.get('duration_months', 0) > 0:
                best_skill = skill.get('name')
                break
        if not best_skill and skills:
            best_skill = skills[0].get('name')

        rare_skills_list = []
        rare_skill_keywords = ['rag', 'fine-tuning', 'lora', 'qlora', 'peft', 'vector database', 'pinecone', 'weaviate', 'qdrant', 'milvus']
        for skill in skills:
            skill_name = skill.get('name', '').lower()
            if any(rare in skill_name for rare in rare_skill_keywords):
                rare_skills_list.append(skill.get('name'))
    else:
        current_title = "Unknown"
        current_company = "Unknown"
        years_of_experience = 0
        location = "Unknown"
        country = "Unknown"
        willing_to_relocate = False
        best_skill = None
        rare_skills_list = []

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

    template_seed = int((score * 1000) + (years_of_experience * 100)) % 5
    
    reasoning = ""

    if tier == 'high':
        if template_seed == 0:
            reasoning = f"{current_title} at {current_company} with {years_of_experience} years of experience"
            if best_skill:
                reasoning += f" demonstrates expertise in {best_skill}"
            if rare_skills_list:
                reasoning += f" with {rare_skills_list[0]} capabilities"
            if semantic_similarity > 0.7:
                reasoning += " and strong contextual alignment with JD requirements"
            if location_fit < 0.6:
                reasoning += f", though based in {location}, {country}"
            reasoning += "."
        elif template_seed == 1:
            reasoning = f"With {years_of_experience} years as {current_title}, this candidate"
            if best_skill:
                reasoning += f" brings strong {best_skill} experience"
            if domain_experience > 0.7:
                reasoning += " from product company environments"
            reasoning += f" at {current_company}"
            if engagement_score > 0.8:
                reasoning += " with high platform engagement"
            if location_fit < 0.6 and willing_to_relocate:
                reasoning += " and is open to relocation"
            reasoning += "."
        elif template_seed == 2:
            reasoning = f"{current_company}'s {current_title} ({years_of_experience} years)"
            if best_skill:
                reasoning += f" has proven {best_skill} capabilities"
            if skill_coherence >= 0.7:
                reasoning += " with solid production deployment evidence"
            if employer_quality > 0.7:
                reasoning += " at high-quality employers"
            if location_fit < 0.6:
                reasoning += f", though located in {location}"
            reasoning += "."
        elif template_seed == 3:
            reasoning = f"Experienced {current_title} with {years_of_experience} years at {current_company}"
            if best_skill:
                reasoning += f", specializing in {best_skill}"
            if production_seniority >= 0.7:
                reasoning += " with proven production ML systems experience"
            if semantic_similarity > 0.6:
                reasoning += " and strong semantic match to role requirements"
            reasoning += "."
        else:
            reasoning = f"{years_of_experience} years as {current_title} at {current_company}"
            if best_skill:
                reasoning += f" with demonstrated {best_skill} expertise"
            if rare_skill_score > 0.5:
                reasoning += " including rare high-value skills"
            if location_fit < 0.6:
                reasoning += f", based in {location}"
            reasoning += "."

    elif tier == 'medium':
        if template_seed == 0:
            reasoning = f"{current_title} with {years_of_experience} years of experience"
            if best_skill:
                reasoning += f" shows {best_skill} skills"
            if semantic_similarity > 0.5:
                reasoning += " with reasonable contextual alignment"
            if location_fit < 0.6:
                reasoning += f", though location in {location} may require relocation"
            if behavioral_multiplier < 0.7:
                reasoning += " with recent inactivity affecting availability"
            reasoning += "."
        elif template_seed == 1:
            reasoning = f"With {years_of_experience} years as {current_title}"
            if best_skill:
                reasoning += f" and {best_skill} experience"
            if skill_coherence < 0.5:
                reasoning += ", though skill coherence needs verification"
            if domain_experience < 0.5:
                reasoning += " with limited product company exposure"
            reasoning += f" from {current_company}."
        elif template_seed == 2:
            reasoning = f"{current_company}'s {current_title} ({years_of_experience} years)"
            if location_fit < 0.6:
                reasoning += f" is based in {location}"
            if production_seniority < 0.5:
                reasoning += " with limited production deployment evidence"
            if engagement_score < 0.5:
                reasoning += " and moderate platform engagement"
            reasoning += "."
        else:
            reasoning = f"{current_title} at {current_company} for {years_of_experience} years"
            if best_skill:
                reasoning += f" brings {best_skill} knowledge"
            if disqualifier_penalty < -0.1:
                reasoning += " with some career pattern concerns"
            if semantic_similarity < 0.4:
                reasoning += " and limited contextual alignment"
            reasoning += "."

    else:
        if template_seed == 0:
            reasoning = f"{current_title} with {years_of_experience} years"
            if title_fit < 0.5:
                reasoning += " has limited role alignment with target AI/ML engineering position"
            if semantic_similarity < 0.3:
                reasoning += " and weak contextual match to JD"
            if location_fit < 0.6:
                reasoning += f" and is based in {location}"
            reasoning += ", making this a borderline fit."
        elif template_seed == 1:
            reasoning = f"With {years_of_experience} years as {current_title}"
            if skill_coherence < 0.3:
                reasoning += ", skill coherence or evidence is limited"
            if behavioral_multiplier < 0.5:
                reasoning += " and recent platform inactivity affects availability"
            if employer_quality < 0.3:
                reasoning += " with limited product company experience"
            if not (skill_coherence < 0.3 or behavioral_multiplier < 0.5):
                reasoning += ", making this a borderline fit"
            reasoning += "."
        elif template_seed == 2:
            reasoning = f"{current_company}'s {current_title} ({years_of_experience} years)"
            if honeypot_flag:
                reasoning += " has profile inconsistencies requiring verification"
            else:
                reasoning += " shows limited alignment with JD requirements"
            if semantic_similarity < 0.3:
                reasoning += " and poor contextual match"
            reasoning += "."
        else:
            reasoning = f"{current_title} with {years_of_experience} years of experience"
            if title_fit < 0.3:
                reasoning += " in unrelated engineering domains"
            if rare_skill_score < 0.2:
                reasoning += " with limited high-value skills"
            reasoning += ", suggesting limited fit."

    if not reasoning or len(reasoning) < 10:
        reasoning = f"{current_title} with {years_of_experience} years shows limited alignment with JD requirements."

    return reasoning
