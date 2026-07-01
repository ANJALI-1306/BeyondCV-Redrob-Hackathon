# Dataset

This project expects the official candidate dataset provided by the Redrob Hackathon organizers.

## Expected File

Place the official dataset in this directory with the following name:

```text
dataset/candidates.jsonl
```

The dataset is **not included** in this repository to keep the repository lightweight and to comply with the competition's dataset distribution policy.

## Running the Ranker

After placing the dataset in the correct location, run:

```bash
python rank.py --candidates dataset/candidates.jsonl --out submission.csv
```

This command will:

1. Load the candidate dataset.
2. Generate any required cached embeddings if they do not already exist.
3. Perform semantic retrieval.
4. Rank candidates using the LightGBM LambdaMART Learning-to-Rank model.
5. Generate `submission.csv`.
6. Automatically perform internal validation and the official submission validation.

## Sample Dataset

A small sample dataset may be included for demonstration or sandbox deployment purposes. For official evaluation, always use the complete `candidates.jsonl` file supplied by the organizers.

## Output

Successful execution produces:

```text
submission.csv
```

containing the Top-100 ranked candidates in the required submission format.
