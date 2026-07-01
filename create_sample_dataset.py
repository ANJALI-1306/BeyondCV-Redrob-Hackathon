from pathlib import Path

INPUT_FILE = Path("dataset/candidates.jsonl")
OUTPUT_FILE = Path("dataset/sample_candidates.jsonl")

NUM_CANDIDATES = 50

count = 0

with INPUT_FILE.open("r", encoding="utf-8") as infile, \
     OUTPUT_FILE.open("w", encoding="utf-8") as outfile:

    for line in infile:
        if line.strip():
            outfile.write(line)
            count += 1

        if count >= NUM_CANDIDATES:
            break

print(f"Created {OUTPUT_FILE}")
print(f"Total candidates copied: {count}")