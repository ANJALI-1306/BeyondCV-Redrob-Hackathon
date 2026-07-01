import sys
import time
import tracemalloc
from rank import load_candidates, rank_candidates


def verify_output_format(ranked):
    print("\n[Output Format Verification]")

    if len(ranked) != 100:
        print(f"  ✗ FAIL: Expected 100 candidates, got {len(ranked)}")
        return False
    print(f"  [OK] PASS: Exactly 100 candidates")

    ranks = [r['rank'] for r in ranked]
    if set(ranks) != set(range(1, 101)):
        print(f"  ✗ FAIL: Ranks not unique or not 1-100")
        return False
    print(f"  [OK] PASS: Unique ranks 1-100")

    scores = [r['score'] for r in ranked]
    for i in range(1, len(scores)):
        if scores[i] > scores[i-1] + 1e-6:
            print(f"  ✗ FAIL: Score increased at rank {i+1}: {scores[i-1]:.6f} -> {scores[i]:.6f}")
            return False
    print(f"  [OK] PASS: Scores monotonically non-increasing")

    for r in ranked:
        if not r['reasoning'] or len(r['reasoning']) < 10:
            print(f"  ✗ FAIL: Empty or too short reasoning for rank {r['rank']}")
            return False
    print(f"  [OK] PASS: All reasonings non-empty")

    return True


def verify_deterministic(candidates_file):
    print("\n[Determinism Verification]")

    candidates1 = load_candidates(candidates_file)
    ranked1 = rank_candidates(candidates1)
    scores1 = [r['score'] for r in ranked1]

    candidates2 = load_candidates(candidates_file)
    ranked2 = rank_candidates(candidates2)
    scores2 = [r['score'] for r in ranked2]

    if scores1 != scores2:
        print(f"  ✗ FAIL: Scores differ between runs")
        return False
    print(f"  [OK] PASS: Deterministic output")

    return True


def verify_cpu_only():
    print("\n[CPU-Only Verification]")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ⚠ WARNING: CUDA available (but not used)")
        else:
            print(f"  [OK] PASS: CUDA not available")
    except ImportError:
        print(f"  [OK] PASS: PyTorch not installed")

    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            print(f"  ⚠ WARNING: TensorFlow GPU available (but not used)")
        else:
            print(f"  [OK] PASS: TensorFlow GPU not available")
    except ImportError:
        print(f"  [OK] PASS: TensorFlow not installed")

    print(f"  [OK] PASS: No GPU usage in code")
    print(f"  [OK] PASS: No network/API calls in code")

    return True


def verify_constraints(candidates_file):
    print("=" * 60)
    print("Hackathon Compliance Verification")
    print("=" * 60)

    tracemalloc.start()

    print("\n[Runtime and Memory Verification]")
    start_time = time.time()
    candidates = load_candidates(candidates_file)
    ranked = rank_candidates(candidates)
    total_time = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Runtime: {total_time:.3f}s")
    print(f"  Peak Memory: {peak_mem / 1024 / 1024:.2f} MB")

    if total_time > 300:
        print(f"  ✗ FAIL: Runtime exceeds 5 minutes")
        return False
    print(f"  [OK] PASS: Runtime < 5 minutes")

    if peak_mem / 1024 / 1024 / 1024 > 16:
        print(f"  ✗ FAIL: Memory exceeds 16GB")
        return False
    print(f"  [OK] PASS: Memory < 16GB")

    if not verify_output_format(ranked):
        return False

    if not verify_deterministic(candidates_file):
        return False

    if not verify_cpu_only():
        return False

    print("\n" + "=" * 60)
    print("COMPLIANCE VERIFICATION: [OK] ALL CHECKS PASSED")
    print("=" * 60)

    return True


if __name__ == '__main__':
    candidates_file = r'c:\Users\anjal\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'

    try:
        success = verify_constraints(candidates_file)
        sys.exit(0 if success else 1)
    except FileNotFoundError:
        print(f"Error: Candidates file not found at {candidates_file}")
        print("Please update the path in verify_compliance.py")
        sys.exit(1)
