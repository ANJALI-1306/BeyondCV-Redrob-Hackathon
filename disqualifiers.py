from typing import Dict, Any, List
from datetime import datetime, timedelta


class DisqualifierScorer:

    def __init__(self):

        self.consulting_keywords = [
            'tcs', 'tata consultancy services',
            'infosys', 'infosys technologies',
            'wipro', 'wipro technologies',
            'accenture',
            'cognizant', 'cts',
            'capgemini',
            'tech mahindra',
            'hcl technologies',
            'mphasis'
        ]

        self.non_nlp_keywords = [
            'computer vision', 'image processing', 'object detection',
            'speech recognition', 'asr', 'tts', 'speech synthesis',
            'robotics', 'robotic process automation', 'rpa',
            'autonomous vehicles', 'self-driving'
        ]

        self.nlp_keywords = [
            'nlp', 'natural language processing', 'text mining',
            'information retrieval', 'search', 'ranking',
            'recommendation', 'recommender system',
            'transformer', 'bert', 'gpt', 'llm', 'rag',
            'vector database', 'pinecone', 'weaviate', 'qdrant'
        ]

        self.framework_keywords = [
            'langchain', 'langchain tutorial',
            'streamlit', 'gradio',
            'hugging face', 'huggingface',
            'fastapi', 'flask', 'django'
        ]
    
    def compute_disqualifier_score(self, candidate: Dict[str, Any]) -> Dict[str, Any]:

        profile = candidate.get('profile', {})
        career = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        signals = candidate.get('redrob_signals', {})

        disqualifiers = {}

        disqualifiers['title_chaser'] = self._detect_title_chaser(career)

        disqualifiers['pure_consulting'] = self._detect_pure_consulting(career)

        disqualifiers['non_nlp_expertise'] = self._detect_non_nlp_expertise(skills)

        disqualifiers['research_only'] = self._detect_research_only(career, profile)

        disqualifiers['inactive'] = self._detect_inactive(signals)

        disqualifiers['weak_progression'] = self._detect_weak_progression(career)

        disqualifiers['domain_mismatch'] = self._detect_domain_mismatch(career)

        disqualifiers['framework_enthusiast'] = self._detect_framework_enthusiast(skills, profile)

        disqualifiers['stagnant_career'] = self._detect_stagnant_career(career)

        disqualifiers['long_notice'] = self._detect_long_notice(signals)

        disqualifiers['no_recent_coding'] = self._detect_no_recent_coding(career, profile)

        weights = {
            'title_chaser': 0.18,
            'pure_consulting': 0.14,
            'non_nlp_expertise': 0.14,
            'research_only': 0.09,
            'inactive': 0.09,
            'weak_progression': 0.09,
            'domain_mismatch': 0.07,
            'framework_enthusiast': 0.05,
            'stagnant_career': 0.04,
            'long_notice': 0.03,
            'no_recent_coding': 0.08,
        }

        composite_score = sum(
            weights[key] * disqualifiers[key]
            for key in weights
        )

        return {
            'disqualifier_score': composite_score,
            'disqualifier_penalty': 1.0 - composite_score,
            'breakdown': disqualifiers
        }

    def _detect_title_chaser(self, career: List[Dict[str, Any]]) -> float:

        if len(career) < 2:
            return 0.0

        short_tenures = 0
        title_progressions = 0

        for i, role in enumerate(career[:-1]):
            duration = role.get('duration_months', 0)
            next_role = career[i + 1]

            if duration > 0 and duration < 18:
                short_tenures += 1

            current_title = role.get('title', '').lower()
            next_title = next_role.get('title', '').lower()

            if ('junior' in current_title and 'senior' in next_title) or \
               ('senior' in current_title and ('staff' in next_title or 'principal' in next_title)):
                title_progressions += 1

        if len(career) == 0:
            return 0.0

        short_tenure_ratio = short_tenures / len(career)

        if short_tenure_ratio > 0.5 and title_progressions > 0:
            return 0.8
        elif short_tenure_ratio > 0.3:
            return 0.5
        else:
            return 0.0
    
    def _detect_pure_consulting(self, career: List[Dict[str, Any]]) -> float:

        if not career:
            return 0.0

        consulting_roles = 0

        for role in career:
            company = role.get('company', '').lower()
            if any(cons in company for cons in self.consulting_keywords):
                consulting_roles += 1

        consulting_ratio = consulting_roles / len(career)

        if consulting_ratio == 1.0:
            return 1.0
        elif consulting_ratio > 0.7:
            return 0.7
        elif consulting_ratio > 0.5:
            return 0.4
        else:
            return 0.0
    
    def _detect_non_nlp_expertise(self, skills: List[Dict[str, Any]]) -> float:

        if not skills:
            return 0.0

        skill_names = [s.get('name', '').lower() for s in skills]

        non_nlp_count = sum(1 for name in skill_names
                          if any(kw in name for kw in self.non_nlp_keywords))

        nlp_count = sum(1 for name in skill_names
                       if any(kw in name for kw in self.nlp_keywords))

        if non_nlp_count > 2 and nlp_count == 0:
            return 0.9
        elif non_nlp_count > 1 and nlp_count == 0:
            return 0.6
        else:
            return 0.0
    
    def _detect_research_only(self, career: List[Dict[str, Any]], profile: Dict[str, Any]) -> float:

        title = profile.get('current_title', '').lower()

        research_keywords = ['research scientist', 'research engineer',
                           'postdoc', 'phd student', 'researcher']

        if any(kw in title for kw in research_keywords):
            production_keywords = ['production', 'deploy', 'mlops', 'infrastructure']
            has_production = False

            for role in career:
                description = role.get('description', '').lower()
                if any(kw in description for kw in production_keywords):
                    has_production = True
                    break

            if not has_production:
                return 0.8
            else:
                return 0.3

        return 0.0
    
    def _detect_inactive(self, signals: Dict[str, Any]) -> float:

        last_active = signals.get('last_active_date', '')
        response_rate = signals.get('recruiter_response_rate', 0)

        if not last_active:
            return 0.5

        try:
            last_active_date = datetime.strptime(last_active, '%Y-%m-%d')
            days_inactive = (datetime.now() - last_active_date).days
        except:
            return 0.5

        if days_inactive > 90 or response_rate < 0.2:
            return 0.9
        elif days_inactive > 60 or response_rate < 0.4:
            return 0.6
        elif days_inactive > 30 or response_rate < 0.6:
            return 0.3
        else:
            return 0.0
    
    def _detect_weak_progression(self, career: List[Dict[str, Any]]) -> float:

        if len(career) < 2:
            return 0.0

        titles = [role.get('title', '').lower() for role in career]

        unique_titles = set(titles)

        if len(unique_titles) == 1:
            return 0.7
        elif len(unique_titles) == 2:
            return 0.3
        else:
            return 0.0
    
    def _detect_domain_mismatch(self, career: List[Dict[str, Any]]) -> float:

        non_tech_industries = [
            'retail', 'manufacturing', 'automotive', 'healthcare',
            'banking', 'finance', 'insurance', 'government'
        ]

        if not career:
            return 0.0

        non_tech_roles = 0
        for role in career:
            industry = role.get('industry', '').lower()
            if any(ind in industry for ind in non_tech_industries):
                non_tech_roles += 1

        non_tech_ratio = non_tech_roles / len(career)

        if non_tech_ratio > 0.8:
            return 0.6
        elif non_tech_ratio > 0.5:
            return 0.3
        else:
            return 0.0
    
    def _detect_framework_enthusiast(self, skills: List[Dict[str, Any]], profile: Dict[str, Any]) -> float:

        skill_names = [s.get('name', '').lower() for s in skills]

        framework_count = sum(1 for name in skill_names
                            if any(kw in name for kw in self.framework_keywords))

        systems_keywords = ['architecture', 'design', 'scalability',
                          'infrastructure', 'distributed']
        summary = profile.get('summary', '').lower()
        has_systems = any(kw in summary for kw in systems_keywords)

        if framework_count > 2 and not has_systems:
            return 0.6
        else:
            return 0.0
    
    def _detect_stagnant_career(self, career: List[Dict[str, Any]]) -> float:

        if not career:
            return 0.0

        for role in career:
            duration = role.get('duration_months', 0)
            if duration > 60:
                title = role.get('title', '').lower()
                if 'junior' in title or 'associate' in title:
                    return 0.7

        return 0.0
    
    def _detect_long_notice(self, signals: Dict[str, Any]) -> float:

        notice = signals.get('notice_period_days', 0)

        if notice > 90:
            return 0.5
        elif notice > 60:
            return 0.3
        else:
            return 0.0
    
    def _detect_no_recent_coding(self, career: List[Dict[str, Any]], profile: Dict[str, Any]) -> float:

        if not career:
            return 0.0

        current_title = profile.get('current_title', '').lower()
        years_of_experience = profile.get('years_of_experience', 0)

        senior_keywords = ['senior', 'staff', 'principal', 'lead', 'architect', 'tech lead']
        is_senior = any(kw in current_title for kw in senior_keywords) or years_of_experience >= 5

        if not is_senior:
            return 0.0

        arch_keywords = ['architect', 'tech lead', 'principal', 'staff', 'head of', 'director', 'vp']
        is_arch_role = any(kw in current_title for kw in arch_keywords)

        if not is_arch_role:
            return 0.0

        coding_keywords = [
            'code', 'programming', 'develop', 'implement', 'python', 'java',
            'javascript', 'typescript', 'golang', 'rust', 'cpp', 'c++',
            'github', 'git', 'pull request', 'commit', 'api', 'backend', 'frontend'
        ]

        if career:
            recent_role = career[0]
            description = recent_role.get('description', '').lower()

            coding_mentions = sum(1 for kw in coding_keywords if kw in description)

            if coding_mentions == 0:
                return 0.7
            elif coding_mentions < 3:
                return 0.4
            else:
                return 0.0

        return 0.0


def get_disqualifier_penalty(candidate: Dict[str, Any]) -> float:
    scorer = DisqualifierScorer()
    result = scorer.compute_disqualifier_score(candidate)
    return result['disqualifier_penalty']


if __name__ == '__main__':
    import json

    with open(r'c:\Users\anjal\Downloads\dataset_extracted\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\sample_candidates.json', 'r') as f:
        candidates = json.load(f)

    scorer = DisqualifierScorer()

    for candidate in candidates[:5]:
        result = scorer.compute_disqualifier_score(candidate)
        print(f"\nCandidate: {candidate['candidate_id']}")
        print(f"Disqualifier Score: {result['disqualifier_score']:.3f}")
        print(f"Penalty Multiplier: {result['disqualifier_penalty']:.3f}")
        print("Breakdown:")
        for key, value in result['breakdown'].items():
            if value > 0:
                print(f"  {key}: {value:.3f}")
