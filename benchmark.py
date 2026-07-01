import time
import tracemalloc
import sys
from rank import load_candidates, rank_candidates


def benchmark_ranking(candidates_file: str):

    print("=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    tracemalloc.start()

    print("\n[1/3] Loading candidates...")
    start_time = time.time()
    candidates = load_candidates(candidates_file)
    load_time = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()

    print(f"  Loaded {len(candidates)} candidates")
    print(f"  Time: {load_time:.3f}s")
    print(f"  Memory: {current_mem / 1024 / 1024:.2f} MB (peak: {peak_mem / 1024 / 1024:.2f} MB)")

    print("\n[2/3] Ranking candidates...")
    start_time = time.time()
    ranked = rank_candidates(candidates)
    rank_time = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()

    print(f"  Ranked {len(ranked)} candidates")
    print(f"  Time: {rank_time:.3f}s")
    print(f"  Memory: {current_mem / 1024 / 1024:.2f} MB (peak: {peak_mem / 1024 / 1024:.2f} MB)")

    total_time = load_time + rank_time

    tracemalloc.stop()

    print("\n[3/3] Summary")
    print("=" * 60)
    print(f"Total candidates processed: {len(candidates)}")
    print(f"Total ranking time: {total_time:.3f}s")
    print(f"Peak memory usage: {peak_mem / 1024 / 1024:.2f} MB")
    print(f"Candidates per second: {len(candidates) / total_time:.0f}")

    print("\nConstraint Compliance:")
    print(f"  Runtime ≤ 5 minutes: {'[OK] PASS' if total_time <= 300 else '✗ FAIL'} ({total_time:.3f}s)")
    print(f"  Memory ≤ 16GB: {'[OK] PASS' if peak_mem / 1024 / 1024 / 1024 <= 16 else '✗ FAIL'} ({peak_mem / 1024 / 1024 / 1024:.2f} GB)")

    return total_time, peak_mem


if __name__ == '__main__':
    candidates_file = r'c:\Users\anjal\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'

    try:
        benchmark_ranking(candidates_file)
    except FileNotFoundError:
        print(f"Error: Candidates file not found at {candidates_file}")
        print("Please update the path in benchmark.py")
        sys.exit(1)
