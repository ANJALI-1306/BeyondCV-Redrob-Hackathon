import json
import numpy as np
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import os
import hashlib


class EmbeddingGenerator:

    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):

        self.model_name = model_name
        self.model = None
        self.cache_dir = 'embeddings_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.checksum_file = os.path.join(self.cache_dir, 'checksums.json')
        self.checksums = self._load_checksums()

    def _load_model(self):

        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def _load_checksums(self) -> Dict[str, str]:

        if os.path.exists(self.checksum_file):
            with open(self.checksum_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_checksums(self):

        with open(self.checksum_file, 'w') as f:
            json.dump(self.checksums, f)

    def _compute_checksum(self, data: str) -> str:

        return hashlib.md5(data.encode('utf-8')).hexdigest()

    def generate_jd_embedding(self, jd_text: str) -> np.ndarray:

        self._load_model()
        embedding = self.model.encode(jd_text, show_progress_bar=False)
        return embedding

    def generate_candidate_embedding(self, candidate: Dict[str, Any]) -> np.ndarray:

        self._load_model()

        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        education = candidate.get('education', [])

        text_parts = []

        current_title = profile.get('current_title', '')
        current_company = profile.get('current_company', '')
        if current_title:
            text_parts.append(f"Current role: {current_title} at {current_company}")

        for role in career_history[:5]:
            title = role.get('title', '')
            company = role.get('company', '')
            description = role.get('description', '')
            if description:
                text_parts.append(f"{title} at {company}: {description}")

        skill_names = [s.get('name', '') for s in skills if s.get('name')]
        if skill_names:
            text_parts.append(f"Skills: {', '.join(skill_names[:20])}")

        for edu in education:
            degree = edu.get('degree', '')
            field = edu.get('field_of_study', '')
            if degree or field:
                text_parts.append(f"Education: {degree} in {field}")

        combined_text = ' '.join(text_parts)

        if not combined_text:
            combined_text = profile.get('anonymized_name', 'Unknown candidate')

        embedding = self.model.encode(combined_text, show_progress_bar=False)
        return embedding

    def generate_career_description_embedding(self, candidate: Dict[str, Any]) -> np.ndarray:

        self._load_model()

        career_history = candidate.get('career_history', [])

        descriptions = []
        for role in career_history[:5]:
            description = role.get('description', '')
            if description:
                descriptions.append(description)

        combined_text = ' '.join(descriptions)

        if not combined_text:
            combined_text = "No career description available"

        embedding = self.model.encode(combined_text, show_progress_bar=False)
        return embedding

    def compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:

        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def save_embedding(self, candidate_id: str, embedding: np.ndarray, source_text: str = None):

        cache_path = os.path.join(self.cache_dir, f"{candidate_id}.npy")
        np.save(cache_path, embedding)

        if source_text:
            checksum = self._compute_checksum(source_text)
            self.checksums[candidate_id] = checksum
            self._save_checksums()

    def save_career_description_embedding(self, candidate_id: str, embedding: np.ndarray, source_text: str = None):

        cache_path = os.path.join(self.cache_dir, f"{candidate_id}_career.npy")
        np.save(cache_path, embedding)

        if source_text:
            checksum_key = f"{candidate_id}_career"
            checksum = self._compute_checksum(source_text)
            self.checksums[checksum_key] = checksum
            self._save_checksums()

    def load_embedding(self, candidate_id: str, source_text: str = None) -> np.ndarray:

        cache_path = os.path.join(self.cache_dir, f"{candidate_id}.npy")
        if os.path.exists(cache_path):

            if source_text and candidate_id in self.checksums:
                current_checksum = self._compute_checksum(source_text)
                if current_checksum != self.checksums[candidate_id]:
                    print(f"Checksum mismatch for {candidate_id}, regenerating...")
                    return None
            return np.load(cache_path)
        return None

    def load_career_description_embedding(self, candidate_id: str, source_text: str = None) -> np.ndarray:

        cache_path = os.path.join(self.cache_dir, f"{candidate_id}_career.npy")
        if os.path.exists(cache_path):
            checksum_key = f"{candidate_id}_career"
            if source_text and checksum_key in self.checksums:
                current_checksum = self._compute_checksum(source_text)
                if current_checksum != self.checksums[checksum_key]:
                    print(f"Checksum mismatch for {candidate_id}_career, regenerating...")
                    return None
            return np.load(cache_path)
        return None

    def save_jd_embedding(self, embedding: np.ndarray):

        cache_path = os.path.join(self.cache_dir, "jd_embedding.npy")
        np.save(cache_path, embedding)

    def load_jd_embedding(self) -> np.ndarray:

        cache_path = os.path.join(self.cache_dir, "jd_embedding.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)
        return None


def precompute_all_embeddings(candidates_file: str, jd_file: str):

    generator = EmbeddingGenerator()

    print("Loading job description...")
    with open(jd_file, 'r', encoding='utf-8') as f:
        jd_text = f.read()

    print("Generating JD embedding...")
    jd_embedding = generator.generate_jd_embedding(jd_text)
    generator.save_jd_embedding(jd_embedding)

    jd_checksum = generator._compute_checksum(jd_text)
    generator.checksums['jd'] = jd_checksum
    generator._save_checksums()

    print(f"JD embedding saved: shape {jd_embedding.shape}")

    print("Loading candidates...")
    candidates = []
    with open(candidates_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    print(f"Generating embeddings for {len(candidates)} candidates...")
    for i, candidate in enumerate(candidates):
        if i % 1000 == 0:
            print(f"Progress: {i}/{len(candidates)}")

        candidate_id = candidate['candidate_id']

        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])
        skills = candidate.get('skills', [])
        education = candidate.get('education', [])

        text_parts = []
        text_parts.append(profile.get('current_title', ''))
        text_parts.append(profile.get('current_company', ''))
        for role in career_history[:5]:
            text_parts.append(role.get('description', ''))
        for skill in skills[:20]:
            text_parts.append(skill.get('name', ''))
        for edu in education:
            text_parts.append(edu.get('degree', ''))

        source_text = ' '.join(text_parts)

        embedding = generator.generate_candidate_embedding(candidate)
        generator.save_embedding(candidate_id, embedding, source_text)

        current_title = profile.get('current_title', '')
        if current_title:
            title_embedding = generator.model.encode(current_title, show_progress_bar=False)
            generator.save_embedding(f"{candidate_id}_title", title_embedding, current_title)

        current_company = profile.get('current_company', '')
        if current_company:
            company_embedding = generator.model.encode(current_company, show_progress_bar=False)
            generator.save_embedding(f"{candidate_id}_company", company_embedding, current_company)

        skill_names = [s.get('name', '') for s in skills[:20] if s.get('name')]
        if skill_names:
            combined_skills = ' '.join(skill_names)
            skill_embedding = generator.model.encode(combined_skills, show_progress_bar=False)
            generator.save_embedding(f"{candidate_id}_skills", skill_embedding, combined_skills)

        fields = []
        for edu in education:
            field = edu.get('field_of_study', '')
            if field:
                fields.append(field)
        if fields:
            combined_fields = ' '.join(fields)
            education_embedding = generator.model.encode(combined_fields, show_progress_bar=False)
            generator.save_embedding(f"{candidate_id}_education", education_embedding, combined_fields)

        career_embedding = generator.generate_career_description_embedding(candidate)
        career_text = ' '.join([role.get('description', '') for role in career_history[:5]])
        generator.save_career_description_embedding(candidate_id, career_embedding, career_text)

    print("All embeddings computed and cached!")
    print(f"Checksums saved to {generator.checksum_file}")


def ensure_embeddings_exist(candidates_file: str, jd_file: str):

    generator = EmbeddingGenerator()

    jd_embedding = generator.load_jd_embedding()
    if jd_embedding is None:
        print("JD embedding missing, generating...")
        with open(jd_file, 'r', encoding='utf-8') as f:
            jd_text = f.read()
        jd_embedding = generator.generate_jd_embedding(jd_text)
        generator.save_jd_embedding(jd_embedding)
        generator.checksums['jd'] = generator._compute_checksum(jd_text)
        generator._save_checksums()
        print("JD embedding generated and cached.")

    print("Embedding cache ready for on-demand generation.")
    print("Run 'python embeddings.py' to pre-compute all embeddings for faster ranking.")


if __name__ == '__main__':

    candidates_file = r'c:\Users\anjal\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'
    jd_file = 'job_description.md'

    precompute_all_embeddings(candidates_file, jd_file)
