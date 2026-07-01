#!/usr/bin/env python3

from validate_submission import validate_submission as official_validate_submission
import argparse
import json
import csv
from typing import Dict, List, Any
import sys

from features import extract_features
from scoring import compute_score
from reasoning import generate_reasoning
from semantic_features import SemanticFeatures
from utils import validate_candidate
from embeddings import ensure_embeddings_exist
import os
LTR_MODEL_PATH = "ltr_model.pkl"

def load_candidates(filepath: str) -> List[Dict[str, Any]]:

    candidates = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    candidate = json.loads(line)
                    validated = validate_candidate(candidate)
                    if validated and validated.get('candidate_id') != 'UNKNOWN':
                        candidates.append(validated)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Warning: Skipping malformed candidate line: {e}")
                    continue
    return candidates


def rank_candidates(candidates: List[Dict[str, Any]], use_ltr: bool = True) -> List[Dict[str, Any]]:

    ltr_model = None
    ltr_feature_names = None

    print("\n" + "=" * 60)
    print("Ranking Engine")
    print("=" * 60)

    if use_ltr and os.path.isfile(LTR_MODEL_PATH):
        try:
           from ltr import load_ltr_model

           ltr_model, ltr_feature_names = load_ltr_model(LTR_MODEL_PATH)

           print("[OK] LightGBM LambdaMART model loaded")
           print(f"[OK] Features Loaded: {len(ltr_feature_names)}")
           print("[OK] Primary Ranking Engine: Learning-to-Rank")

        except Exception as e:
           print(f"\nWARNING: Failed to load LTR model: {e}")
           print("Switching to deterministic manual scoring.")
    else:
        print("LTR model not found.")
        print("Using deterministic manual scoring.")

    print("Stage 1: Embedding-based retrieval...")
    try:
        semantic = SemanticFeatures()
        candidates_with_similarity = []

        for candidate in candidates:
            try:
                similarity = semantic.compute_semantic_similarity(candidate)
                candidates_with_similarity.append({
                    'candidate': candidate,
                    'semantic_similarity': similarity
                })
            except Exception:
                candidates_with_similarity.append({
                    'candidate': candidate,
                    'semantic_similarity': 0.0
                })

        candidates_with_similarity.sort(key=lambda x: x['semantic_similarity'], reverse=True)
        top_500 = candidates_with_similarity[:500]
        print(f"Stage 1 complete: Retrieved {len(top_500)} candidates")

    except (ValueError, FileNotFoundError):
        print("Embeddings not available, using all candidates for Stage 1")
        top_500 = [{'candidate': c, 'semantic_similarity': 0.0} for c in candidates]

    print("Stage 2: Full feature scoring...")
    ranked = []

    for item in top_500:
        candidate = item['candidate']

        features = extract_features(candidate)

        score = compute_score(features, candidate['candidate_id'], ltr_model, ltr_feature_names)

        ranked.append({
            'candidate_id': candidate['candidate_id'],
            'candidate': candidate,
            'features': features,
            'score': score,
            'semantic_similarity': item['semantic_similarity']
        })

    ranked.sort(key=lambda x: (-x['score'], x['candidate_id']))

    for idx, item in enumerate(ranked[:100]):
        rank = idx + 1
        item['rank'] = rank

        if rank <= 33:
            tier = 'high'
        elif rank <= 66:
            tier = 'medium'
        else:
            tier = 'low'

        reasoning = generate_reasoning(
            item['features'],
            item['score'],
            item['candidate'],
            tier
        )
        item['reasoning'] = reasoning

    return [{
        'candidate_id': item['candidate_id'],
        'rank': item['rank'],
        'score': item['score'],
        'reasoning': item['reasoning']
    } for item in ranked[:100]]


def write_csv(ranked_candidates: List[Dict[str, Any]], output_path: str):

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])

        for candidate in ranked_candidates:
            writer.writerow([
                candidate['candidate_id'],
                candidate['rank'],
                f"{candidate['score']:.6f}",
                candidate['reasoning']
            ])


def validate_internal_submission(ranked_candidates: List[Dict[str, Any]]) -> bool:

    if len(ranked_candidates) != 100:
        print(f"ERROR: Expected 100 candidates, got {len(ranked_candidates)}")
        return False

    ranks = [r['rank'] for r in ranked_candidates]
    if set(ranks) != set(range(1, 101)):
        print(f"ERROR: Ranks not unique or not 1-100")
        return False

    scores = [r['score'] for r in ranked_candidates]
    for i in range(1, len(scores)):
        if scores[i] > scores[i-1] + 1e-6:
            print(f"ERROR: Score increased at rank {i+1}: {scores[i-1]:.6f} -> {scores[i]:.6f}")
            return False

    for r in ranked_candidates:
        if not r['reasoning'] or len(r['reasoning']) < 10:
            print(f"ERROR: Empty or too short reasoning for rank {r['rank']}")
            return False

    candidate_ids = [r['candidate_id'] for r in ranked_candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        print(f"ERROR: Duplicate candidate IDs found")
        return False

    print("[OK] Internal validation passed")
    return True


def main():
    parser = argparse.ArgumentParser(description='Rank candidates for Senior AI Engineer role')
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl')
    parser.add_argument('--out', required=True, help='Path to output CSV')
    parser.add_argument('--jd', default='job_description.md', help='Path to job description file')
    parser.add_argument(
       "--sandbox",
       action="store_true",
       help="Run in sandbox mode (skip official validation)"
    )
    args = parser.parse_args()

    print("Checking embedding cache...")
    ensure_embeddings_exist(args.candidates, args.jd)

    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates")

    print("Ranking candidates...")
    ranked = rank_candidates(candidates)
    print(f"Generated Top-{len(ranked)} candidates")
    print(f"Top Candidate : {ranked[0]['candidate_id']}")
    print(f"Top Score     : {ranked[0]['score']:.6f}")
    print(f"Ranked {len(ranked)} candidates")

    print(f"Writing output to {args.out}...")
    write_csv(ranked, args.out)

    print("\n" + "=" * 60)
    print("Internal Validation")
    print("=" * 60)

    if args.sandbox:
        print("[OK] Sandbox mode - skipping Top-100 validation.")

    else:
        if not validate_internal_submission(ranked):
            print("ERROR: Internal validation failed.")
            sys.exit(1)

        print("[OK] Internal validation passed")

    if args.sandbox:
        print("\n" + "=" * 60)
        print("Sandbox Mode")
        print("=" * 60)
        print("Sandbox mode enabled.")
        print("Skipping official submission validator.")
    else:
        print("\n" + "=" * 60)
        print("Official Redrob Submission Validation")
        print("=" * 60)

        errors = official_validate_submission(args.out)

        if errors:
            print(f"\nValidation failed ({len(errors)} issue(s)):\n")

            for error in errors:
                print(f"  • {error}")

            sys.exit(1)

        print("[OK] Official validator passed")

    print("\nSubmission Ready!")

if __name__ == "__main__":
    main()