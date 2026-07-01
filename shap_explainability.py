import numpy as np
import pickle
from typing import Dict, Any, List
import shap


class SHAPExplainer:

    def __init__(self, model_path: str = 'ltr_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        self.feature_names = None

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Loaded LTR model from {self.model_path}")
        except FileNotFoundError:
            print(f"LTR model not found at {self.model_path}")
            raise

    def get_feature_names(self) -> List[str]:
        if self.model is None:
            self.load_model()

        if hasattr(self.model, 'feature_name'):
            return self.model.feature_name()
        else:
            return [f'feature_{i}' for i in range(100)]

    def create_explainer(self, background_data: np.ndarray):
        if self.model is None:
            self.load_model()

        self.feature_names = self.get_feature_names()

        self.explainer = shap.TreeExplainer(self.model, background_data)
        print("SHAP explainer created")

    def explain_candidate(self, features: Dict[str, float]) -> Dict[str, Any]:
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")

        feature_array = self._features_to_array(features)

        shap_values = self.explainer.shap_values(feature_array)

        explanation = {
            'shap_values': shap_values[0].tolist(),
            'feature_names': self.feature_names,
            'base_value': float(self.explainer.expected_value),
            'prediction': float(self.model.predict(feature_array.reshape(1, -1))[0]),
            'feature_importance': self._get_feature_importance(shap_values[0]),
        }

        return explanation

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        if self.feature_names is None:
            self.feature_names = self.get_feature_names()

        array = []
        for name in self.feature_names:
            array.append(features.get(name, 0.0))

        return np.array(array)

    def _get_feature_importance(self, shap_values: np.ndarray) -> List[Dict[str, float]]:
        importance = []
        for i, (name, value) in enumerate(zip(self.feature_names, shap_values)):
            importance.append({
                'feature': name,
                'shap_value': float(value),
                'abs_importance': abs(float(value))
            })

        importance.sort(key=lambda x: x['abs_importance'], reverse=True)

        return importance

    def generate_explanation_text(self, explanation: Dict[str, Any], top_n: int = 5) -> str:
        lines = []
        lines.append(f"Prediction score: {explanation['prediction']:.4f}")
        lines.append(f"Base value: {explanation['base_value']:.4f}")
        lines.append("\nTop contributing features:")

        for i, feat in enumerate(explanation['feature_importance'][:top_n]):
            direction = "+" if feat['shap_value'] > 0 else "-"
            lines.append(f"  {i+1}. {feat['feature']}: {direction}{abs(feat['shap_value']):.4f}")

        return "\n".join(lines)


def explain_ranking(ranked_candidates: List[Dict[str, Any]],
                   background_data: np.ndarray = None,
                   model_path: str = 'ltr_model.pkl') -> Dict[str, Any]:
    explainer = SHAPExplainer(model_path)

    if background_data is None:
        feature_dicts = [c['features'] for c in ranked_candidates[:100]]
        background_data = np.array([
            [feat.get(name, 0.0) for name in explainer.get_feature_names()]
            for feat in feature_dicts
        ])

    explainer.create_explainer(background_data)

    explanations = {}
    for candidate in ranked_candidates[:10]:
        candidate_id = candidate['candidate_id']
        features = candidate['features']

        explanation = explainer.explain_candidate(features)
        explanation_text = explainer.generate_explanation_text(explanation)

        explanations[candidate_id] = {
            'explanation': explanation,
            'explanation_text': explanation_text
        }

    return explanations


if __name__ == '__main__':
    import json
    from rank import load_candidates, rank_candidates

    print("Loading candidates...")
    candidates_file = r'c:\Users\anjal\Downloads\dataset_extracted\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'
    candidates = load_candidates(candidates_file)

    print("Ranking candidates...")
    ranked = rank_candidates(candidates)

    print("Generating SHAP explanations...")
    explanations = explain_ranking(ranked)

    print("\nSHAP explanations for top 3 candidates:")
    for i, (candidate_id, expl) in enumerate(explanations.items()):
        if i >= 3:
            break
        print(f"\n{candidate_id}:")
        print(expl['explanation_text'])
