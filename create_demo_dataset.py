from pathlib import Path

INPUT_FILE = Path("dataset/candidates.jsonl")
OUTPUT_FILE = Path("dataset/demo_candidates.jsonl")

NUM_CANDIDATES = 100

count = 0

with INPUT_FILE.open("r", encoding="utf-8") as infile, \
     OUTPUT_FILE.open("w", encoding="utf-8") as outfile:

    for line in infile:
        if line.strip():
            outfile.write(line)
            count += 1

        if count >= NUM_CANDIDATES:
            break

print("=" * 50)
print("Demo dataset created successfully!")
print(f"Output File : {OUTPUT_FILE}")
print(f"Candidates  : {count}")
print("=" * 50)