import numpy as np
from typing import Dict, Any
from embeddings import EmbeddingGenerator


class SemanticFeatures:

    def __init__(self, embedding_generator: EmbeddingGenerator = None):
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.jd_embedding = None

    def _load_jd_embedding(self):
        if self.jd_embedding is None:
            self.jd_embedding = self.embedding_generator.load_jd_embedding()
            if self.jd_embedding is None:
                raise ValueError("JD embedding not found. Run embeddings.py first to pre-compute.")

    def compute_semantic_similarity(self, candidate: Dict[str, Any]) -> float:
        self._load_jd_embedding()

        candidate_id = candidate['candidate_id']
        candidate_embedding = self.embedding_generator.load_embedding(candidate_id)

        if candidate_embedding is None:
            return 0.0

        similarity = self.embedding_generator.compute_cosine_similarity(
            self.jd_embedding, candidate_embedding
        )

        return float(similarity)

    def compute_title_semantic_similarity(self, candidate: Dict[str, Any]) -> float:
        self._load_jd_embedding()

        profile = candidate.get('profile', {})
        current_title = profile.get('current_title', '')

        if not current_title:
            return 0.0

        candidate_id = candidate['candidate_id']
        title_embedding = self.embedding_generator.load_embedding(f"{candidate_id}_title")

        if title_embedding is None:
            return 0.0

        similarity = self.embedding_generator.compute_cosine_similarity(
            self.jd_embedding, title_embedding
        )

        return float(similarity)

    def compute_skill_jd_overlap(self, candidate: Dict[str, Any]) -> float:
        profile = candidate.get('profile', {})
        skills = candidate.get('skills', [])

        jd_keywords = [
            'python', 'machine learning', 'ml', 'deep learning', 'pytorch', 'tensorflow',
            'scikit-learn', 'numpy', 'pandas',
            'embeddings', 'vector database', 'pinecone', 'weaviate', 'qdrant', 'milvus',
            'elasticsearch', 'opensearch', 'faiss',
            'retrieval', 'ranking', 'recommendation', 'search',
            'llm', 'langchain', 'rag', 'fine-tuning', 'nlp',
            'airflow', 'spark', 'kafka', 'sql',
            'lora', 'qlora', 'peft'
        ]

        if not skills:
            return 0.0

        matching_skills = 0
        for skill in skills:
            skill_name = skill.get('name', '').lower()
            if any(keyword in skill_name for keyword in jd_keywords):
                matching_skills += 1

        overlap_score = matching_skills / len(skills) if skills else 0.0

        return float(overlap_score)

    def compute_all_semantic_features(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        features = {
            'semantic_similarity': self.compute_semantic_similarity(candidate),
            'title_semantic_similarity': self.compute_title_semantic_similarity(candidate),
            'skill_jd_overlap': self.compute_skill_jd_overlap(candidate),
            'career_description_semantic_similarity': self.compute_career_description_similarity(candidate),
            'company_name_semantic_similarity': self.compute_company_name_similarity(candidate),
            'skill_name_semantic_similarity': self.compute_skill_name_similarity(candidate),
            'education_field_semantic_similarity': self.compute_education_field_similarity(candidate),
        }

        return features

    def compute_career_description_similarity(self, candidate: Dict[str, Any]) -> float:
        self._load_jd_embedding()

        candidate_id = candidate['candidate_id']
        career_embedding = self.embedding_generator.load_career_description_embedding(candidate_id)

        if career_embedding is None:
            return 0.0

        similarity = self.embedding_generator.compute_cosine_similarity(
            self.jd_embedding, career_embedding
        )

        return float(similarity)

    def compute_company_name_similarity(self, candidate: Dict[str, Any]) -> float:
        self._load_jd_embedding()

        profile = candidate.get('profile', {})
        current_company = profile.get('current_company', '')

        if not current_company:
            return 0.0

        candidate_id = candidate['candidate_id']
        company_embedding = self.embedding_generator.load_embedding(f"{candidate_id}_company")

        if company_embedding is None:
            return 0.0

        similarity = self.embedding_generator.compute_cosine_similarity(
            self.jd_embedding, company_embedding
        )

        return float(similarity)

    def compute_skill_name_similarity(self, candidate: Dict[str, Any]) -> float:
        self._load_jd_embedding()

        candidate_id = candidate['candidate_id']
        skill_embedding = self.embedding_generator.load_embedding(f"{candidate_id}_skills")

        if skill_embedding is None:
            return 0.0

        similarity = self.embedding_generator.compute_cosine_similarity(
            self.jd_embedding, skill_embedding
        )

        return float(similarity)

    def compute_education_field_similarity(self, candidate: Dict[str, Any]) -> float:
        self._load_jd_embedding()

        candidate_id = candidate['candidate_id']
        education_embedding = self.embedding_generator.load_embedding(f"{candidate_id}_education")

        if education_embedding is None:
            return 0.0

        similarity = self.embedding_generator.compute_cosine_similarity(
            self.jd_embedding, education_embedding
        )

        return float(similarity)
