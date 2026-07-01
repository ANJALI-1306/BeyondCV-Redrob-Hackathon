import numpy as np
import lightgbm as lgb
from typing import Dict, List, Any, Tuple
import pickle
import os


def generate_pseudo_labels(candidates: List[Dict[str, Any]], ranked: List[Dict[str, Any]]) -> Dict[str, int]:

    rank_map = {item['candidate_id']: item['rank'] for item in ranked}

    pseudo_labels = {}
    for candidate in candidates:
        candidate_id = candidate['candidate_id']
        rank = rank_map.get(candidate_id, 101)

        if rank <= 10:
            relevance = 4
        elif rank <= 30:
            relevance = 3
        elif rank <= 60:
            relevance = 2
        elif rank <= 100:
            relevance = 1
        else:
            relevance = 0

        pseudo_labels[candidate_id] = relevance

    return pseudo_labels


def prepare_ltr_data(candidates: List[Dict[str, Any]], pseudo_labels: Dict[str, int], max_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    from features import extract_features

    sorted_candidates = sorted(candidates, key=lambda c: pseudo_labels.get(c['candidate_id'], 0), reverse=True)
    candidates = sorted_candidates[:max_samples]

    feature_dicts = []
    labels = []

    for candidate in candidates:
        candidate_id = candidate['candidate_id']
        features = extract_features(candidate)

        numeric_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                numeric_features[key] = value
        
        feature_dicts.append(numeric_features)
        labels.append(pseudo_labels[candidate_id])

    feature_names = list(feature_dicts[0].keys()) if feature_dicts else []
    X = np.array([[f[name] for name in feature_names] for f in feature_dicts])
    y = np.array(labels)
    qids = np.zeros(len(candidates), dtype=np.int32)

    return X, y, qids, feature_names


def train_ltr_model(X: np.ndarray, y: np.ndarray, qids: np.ndarray, feature_names: List[str]) -> lgb.LGBMRanker:

    train_data = lgb.Dataset(X, label=y, group=[len(X)])

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10, 50],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(period=10)]
    )

    model.feature_name = feature_names

    return model


def save_ltr_model(model: lgb.LGBMRanker, feature_names: List[str], filepath: str = 'ltr_model.pkl'):

    model_data = {
        'model': model,
        'feature_names': feature_names,
    }

    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"LTR model saved to {filepath}")


def load_ltr_model(filepath: str = 'ltr_model.pkl') -> Tuple[lgb.LGBMRanker, List[str]]:

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"LTR model not found at {filepath}")

    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)

    return model_data['model'], model_data['feature_names']


def predict_ltr_score(model: lgb.LGBMRanker, features: Dict[str, Any], feature_names: List[str]) -> float:

    feature_vector = np.array([features.get(name, 0.0) for name in feature_names])

    feature_vector = feature_vector.reshape(1, -1)

    score = model.predict(feature_vector)[0]

    return float(score)


def get_feature_importance(model: lgb.LGBMRanker) -> Dict[str, float]:

    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name

    importance_dict = dict(zip(feature_names, importance))

    total = sum(importance_dict.values())
    if total > 0:
        importance_dict = {k: v / total for k, v in importance_dict.items()}

    return importance_dict


def train_ltr_pipeline(candidates: List[Dict[str, Any]], ranked: List[Dict[str, Any]],
                       model_path: str = 'ltr_model.pkl') -> lgb.LGBMRanker:

    print("Generating pseudo-labels from current ranking...")
    pseudo_labels = generate_pseudo_labels(candidates, ranked)

    print("Preparing LTR data...")
    X, y, qids, feature_names = prepare_ltr_data(candidates, pseudo_labels)

    print(f"Training LTR model on {len(candidates)} candidates with {len(feature_names)} features...")
    model = train_ltr_model(X, y, qids, feature_names)

    print("Saving LTR model...")
    save_ltr_model(model, feature_names, model_path)

    importance = get_feature_importance(model)
    print("\nTop 10 Feature Importance:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, imp in sorted_importance[:10]:
        print(f"  {feature}: {imp:.4f}")

    return model



