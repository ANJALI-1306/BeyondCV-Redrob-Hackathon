# Candidate Ranking System for Redrob AI Recruiter Hackathon

A production-grade AI recruiting system with semantic understanding, expanded feature engineering (70+ features), Learning-to-Rank, and two-stage retrieval for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Overview

Ranks 100,000 candidate profiles for a Senior AI Engineer — Founding Team role at an Indian Series A startup. The system uses explainable, JD-grounded features with semantic embeddings and LightGBM LambdaMART to avoid common traps like keyword stuffing and title-chasing while maximizing NDCG, MAP, and precision metrics.

## Architecture

### Pipeline Architecture

```
┌─────────────────┐
│  Load Candidates │
│   (JSONL)        │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Validate &     │
│  Clean Data     │
│  (utils.py)     │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Stage 1:       │
│  Embedding       │
│  Retrieval       │
│  (Top 500)      │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Stage 2:       │
│  Full Feature   │
│  Scoring        │
│  (Top 100)      │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Generate       │
│  Reasoning      │
│  (Tier-based)   │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Output CSV     │
│  (100 ranked)   │
└─────────────────┘
```

### Module Structure

- **rank.py**: Entry point with CLI args, two-stage retrieval orchestration, LTR model integration
- **features.py**: Feature extraction (70+ explainable features)
- **scoring.py**: Weighted scoring with LTR model support and manual weights fallback
- **reasoning.py**: Programmatic reasoning generation with contextual depth
- **semantic_features.py**: Semantic similarity using pre-computed embeddings (7 features)
- **embeddings.py**: Offline embedding generation with checksum validation (BAAI/bge-small-en-v1.5)
- **ltr.py**: LightGBM LambdaMART training and inference
- **evaluation.py**: Ranking metrics (NDCG, MAP, Precision) with validation dataset support
- **disqualifiers.py**: Dedicated disqualifier scoring module (11 disqualifier types)
- **shap_explainability.py**: SHAP explainability for interview defense
- **feature_ablation.py**: Feature ablation studies for impact analysis
- **validation_dataset_builder.py**: Manual validation dataset builder
- **create_heuristic_labels.py**: Heuristic label generation for evaluation
- **measure_honeypot_rate.py**: Honeypot rate measurement tool
- **utils.py**: Robustness utilities for missing values and edge cases
- **tests/**: Unit tests for feature extraction

### Project Structure

REDROB/ │
├── dataset/
└── candidates.jsonl
│
├── rank.py
├── features.py
├── semantic_features.py
├── scoring.py
├── reasoning.py
├── embeddings.py
├── ltr.py
├── ltr_model.pkl
├── evaluation.py
├── benchmark.py
├── requirements.txt
├── README.md
├── Dockerfile
├── submission_metadata.yaml
└── tests/

## Feature Engineering (70+ Features)

### Core Features (8)

1. **title_role_fit_score** (0-1): Classifies titles into Tier A (core AI/ML/ranking) / Tier B (adjacent) / Tier C (unrelated) / Tier D (disqualifiers)
2. **experience_fit_score** (0-1): Distance from JD's 5-9 year band, lenient outside it
3. **skill_coherence_score** (0-1): Weights skills by proficiency, duration, and career history cross-reference
4. **production_seniority_score** (0-1): Evidence of production deployment, scale, named retrieval/vector-DB tech
5. **location_fit_score** (0-1): India preferred cities > willing to relocate > no relocation signal
6. **disqualifier_penalty** (negative): Title-chasing, pure-consulting-only, CV/speech/robotics-only patterns
7. **behavioral_multiplier** (0.3-1.0): From Redrob signals — activity, response rate, availability
8. **honeypot_flag** (bool) + **honeypot_penalty_multiplier** (float): Flags suspicious profiles

### Semantic Features (7)

9. **semantic_similarity** (0-1): Cosine similarity between candidate and JD embeddings
10. **title_semantic_similarity** (0-1): Semantic match of title to JD
11. **skill_jd_overlap** (0-1): Semantic overlap of skills with JD keywords
12. **career_description_semantic_similarity** (0-1): Career descriptions vs JD
13. **company_name_semantic_similarity** (0-1): Company name relevance
14. **skill_name_semantic_similarity** (0-1): Skill names vs JD
15. **education_field_semantic_similarity** (0-1): Education field relevance

### Experience Features (5)

16. **total_years**: Total years of experience
17. **relevant_years**: Years in AI/ML roles
18. **seniority_level**: Seniority score (0-1)
19. **domain_experience**: Product vs consulting company history
20. **management_experience**: Management/leadership experience score

### Skill Features (5)

21. **skill_count**: Total number of skills
22. **expert_skill_count**: Number of expert-level skills
23. **skill_diversity**: Diversity of skill categories
24. **skill_recency**: Recency of skill usage
25. **rare_skill_score**: Score for rare/high-value skills (RAG, fine-tuning, etc.)

### Career Features (6)

26. **promotion_frequency**: How often candidate was promoted
27. **average_tenure**: Average tenure per role (months)
28. **career_growth_velocity**: Rate of title progression
29. **career_consistency**: Consistency of career path
30. **employer_quality**: Quality of employers (product vs consulting)
31. **leadership_progression**: Progression into leadership roles

### Education Features (5)

32. **degree_relevance**: Relevance of degree to AI/ML
33. **education_quality**: Quality of institution
34. **certification_count**: Number of certifications
35. **academic_consistency**: Consistency of education timeline
36. **education_tier_score**: Average tier of institutions (tier_1=1.0, tier_4=0.25)

### Behavioral Features (6)

37. **profile_completeness**: How complete the profile is
38. **profile_freshness**: How recently profile was updated
39. **recent_activity_score**: Recent platform activity
40. **career_momentum**: Career progression momentum
41. **engagement_score**: Overall engagement level
42. **employment_stability**: Stability of employment history

### Resume Quality Features (3)

43. **profile_completeness_score**: Overall completeness
44. **missing_info_count**: Count of missing important fields
45. **project_richness**: Richness of project descriptions

### Redrob Extended Features (14)

46. **redrob_profile_completeness_score**: Platform's completeness metric (0-100)
47. **profile_visibility_score**: Based on views and search appearances
48. **application_activity_score**: Based on applications submitted
49. **response_speed_score**: Based on avg response time
50. **skill_assessment_avg_score**: Average of skill assessment scores
51. **social_proof_score**: Based on connections and endorsements
52. **availability_score**: Based on notice period
53. **salary_alignment_score**: Alignment with expected salary range
54. **work_mode_match_score**: Match with preferred work mode
55. **github_activity_normalized**: GitHub activity score normalized
56. **recruiter_interest_score**: Based on saves by recruiters
57. **offer_history_score**: Based on offer acceptance rate
58. **verification_score**: Based on verified email/phone/linkedin
59. **platform_engagement_composite**: Composite engagement metric

### Certification Features (4)

60. **certification_count**: Total number of certifications
61. **certification_relevance_score**: Relevance to AI/ML
62. **certification_recency_score**: How recent certifications are
63. **certification_quality_score**: Quality of certifying organizations

### Language Features (4)

64. **language_count**: Total number of languages
65. **english_proficiency_score**: English proficiency level
66. **native_language_count**: Number of native languages
67. **professional_language_count**: Number of professional languages

### Company Size Features (3)

68. **company_size_progression**: Progression from small to large companies
69. **current_company_size_score**: Score based on current company size
70. **avg_company_size_score**: Average company size across career

### Industry Transition Features (3)

71. **industry_consistency**: How consistent the industry has been
72. **industry_diversity**: Number of different industries
73. **current_industry_relevance**: Relevance of current industry to AI/ML

### Honeypot Anomaly Feature (1)

74. **honeypot_anomaly_score**: Statistical anomaly score for honeypot detection

## Scoring

### Learning-to-Rank (Preferred)

The system uses LightGBM LambdaMART for ranking when a trained model is available:

- Trained on pseudo-labels derived from current ranking
- Uses all 70+ features for optimal discrimination
- Provides feature importance for explainability
- Falls back to manual weights if model unavailable
- If ltr_model.pkl is unavailable, the system automatically falls back to deterministic manual feature weighting. This guarantees reproducible execution even without the trained ranking model.

### Manual Weights (Fallback)

If LTR model is not available, uses weighted combination:

**Core Features (0.51 total):**

- title_role_fit: 0.25
- skill_coherence: 0.20
- production_seniority: 0.15
- experience_fit: 0.10
- location_fit: 0.01

**Semantic Features (0.15 total):**

- semantic_similarity: 0.10
- skill_jd_overlap: 0.05

**Expanded Features (0.17 total):**

- domain_experience: 0.05
- employer_quality: 0.03
- engagement_score: 0.03
- rare_skill_score: 0.02
- skill_diversity: 0.02
- career_consistency: 0.02

**Penalties:**

- disqualifier_penalty: additive (negative)

**Multipliers:**

- behavioral_multiplier: 0.3-1.0
- honeypot_penalty_multiplier: graduated penalty (0.05-1.0) based on anomaly score

### Final Score

```
base = weighted_sum(features) + disqualifier_penalty
base_clipped = clip(base, 0, 1)
final_score = base_clipped * behavioral_multiplier * honeypot_penalty_multiplier
```

## Two-Stage Retrieval

### Stage 1: Embedding Retrieval (Top 500)

- Uses pre-computed BAAI/bge-small-en-v1.5 embeddings
- Computes cosine similarity to JD embedding
- Returns top 500 candidates by semantic similarity
- Fallback to all candidates if embeddings not available

### Stage 2: Full Feature Scoring (Top 100)

- Extracts 70+ features for top 500 candidates
- Computes scores using LTR model or manual weights
- Sorts by score with tie-breaking per submission spec (candidate_id ascending for equal scores)
- Returns top 100 ranked candidates

### Tie-Breaking

Per submission specification: if scores are equal, candidates are sorted by candidate_id in ascending order.

## Honeypot Detection

Enhanced detection with statistical anomaly scoring:

- **Hard patterns**: 8 rule-based patterns (expert skills with zero duration, experience mismatch, invalid dates, etc.)
- **Statistical anomaly**: Composite score based on 7 features (skill count, expert ratio, experience mismatch, duration inconsistency, profile completeness, behavioral anomaly, education anomaly)
- **Graduated penalty**: 0.05 for clear honeypots, 0.2-0.8 graduated penalty based on anomaly score, 1.0 for clean profiles

## Reasoning Generation

### Tier-Based Tone

- **High tier (ranks 1-33)**: Glowing, confident tone
- **Medium tier (ranks 34-66)**: Balanced, measured tone
- **Low tier (ranks 67-100)**: Marginal, hedged tone

### Richness Improvements

- Includes semantic similarity mentions
- Cites rare high-value skills (RAG, fine-tuning, etc.)
- References domain experience (product vs consulting)
- Mentions employer quality
- Acknowledges engagement score
- Varies sentence structure across candidates

## Usage

### Basic Ranking

```bash
python rank.py --candidates dataset/candidates.jsonl --out submission.csv
```

### Pre-compute Embeddings (Offline)

```bash
python embeddings.py
```

This generates embeddings for all candidates and JD, caching them to disk for fast loading during ranking.

### Train LTR Model

```bash
python ltr.py
```

This trains a LightGBM LambdaMART model using pseudo-labels from the current ranking and saves it as `ltr_model.pkl`. The model will be automatically loaded during ranking if available.

### Evaluate Ranking

```bash
python evaluation.py
```

This generates an evaluation report with NDCG@10, NDCG@50, MAP, and Precision@10 metrics. Use `--validation validation_labeled.json` to evaluate against the heuristic validation dataset.

### Build Validation Dataset

```bash
python validation_dataset_builder.py --candidates dataset/candidates.jsonl --n 200
```

Creates a validation template for manual labeling.

### Create Heuristic Labels

```bash
python create_heuristic_labels.py --template validation_template.json --output validation_labeled.json
```

Generates heuristic labels for evaluation without manual labeling.

### Measure Honeypot Rate

```bash
python measure_honeypot_rate.py --candidates dataset/candidates.jsonl
```

Measures the percentage of honeypots in the top 100 ranking.

### Feature Ablation Study

```bash
python feature_ablation.py --candidates dataset/candidates.jsonl --validation validation_labeled.json
```

Performs ablation studies to measure feature group impact on ranking quality.

### SHAP Explainability

```bash
python shap_explainability.py
```

Generates SHAP values for LTR model predictions (requires trained ltr_model.pkl).

## Validation

The ranking pipeline performs two levels of validation automatically.

### Internal Validation

The generated submission is checked for:

- Exactly 100 candidates
- Unique candidate IDs
- Ranks 1–100
- Monotonically decreasing scores
- Non-empty reasoning
- Correct CSV format

### Official Validation

The project automatically executes the official Redrob validation script.
Submission completes only after both validation stages pass successfully.

## Constraints Compliance

- **Runtime**: ≤5 minutes wall-clock (tested on 100K candidates)
- **RAM**: ≤16GB (typical usage ~2-4GB)
- **CPU only**: No GPU, no network/API calls during ranking
- **Deterministic**: Same input produces same output
- **Offline**: All computation local, no external dependencies during ranking
- No internet access is required during ranking.
- Hugging Face models are downloaded only once during the first embedding generation and cached locally.
- Subsequent executions use cached embeddings for fully offline ranking.

## Dependencies

```
# Core ML
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
shap>=0.42.0

# Embeddings / NLP
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0
huggingface_hub>=0.20.0
tokenizers>=0.15.0
safetensors>=0.4.0
tqdm>=4.65.0

# Utilities
joblib>=1.3.0
PyYAML>=6.0
python-docx>=0.8.11
```

## Testing

Run unit tests on feature extraction:

```bash
python -m pytest tests/
```

## Output Format

CSV with header: `candidate_id,rank,score,reasoning`

Exactly 100 data rows, ranks 1-100 used exactly once, score non-increasing by rank.

## Performance Benchmarks

### Runtime (100K candidates)

- Without embeddings: ~0.05 seconds
- With pre-computed embeddings: ~2-3 seconds (Stage 1 retrieval)
- Total ranking: ~3-5 seconds

### Memory Usage

- Baseline: ~500MB
- With embeddings loaded: ~2-4GB

### Scalability

- Two-stage retrieval enables scaling to millions of candidates
- Stage 1 reduces feature extraction from 100K to 500 candidates
- Stage 2 applies full scoring only to top 500

## Reproducibility

### Docker Instructions

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python rank.py --candidates dataset/candidates.jsonl --out submission.csv"]
```

### One-Command Execution

```bash
python rank.py --candidates dataset/candidates.jsonl --out submission.csv
```

## Key Improvements Over Baseline

1. **Semantic Understanding**: Added embeddings for contextual matching beyond keywords
2. **Feature Expansion**: Expanded from 8 to 70+ features for better discrimination
3. **Learning-to-Rank**: Implemented LightGBM LambdaMART with pseudo-labels.If ltr_model.pkl is unavailable, the system automatically falls back to deterministic manual feature weighting. This guarantees reproducible execution even without the trained ranking model.
4. **Two-Stage Retrieval**: Improved scalability and performance
5. **Enhanced Honeypot Detection**: Statistical anomaly scoring + 8 rule-based patterns
6. **Richer Reasoning**: Includes semantic, rare skills, domain experience
7. **Better Tie-Breaking**: Per submission spec (candidate_id ascending for equal scores)
8. **Robustness**: Handles missing values and edge cases gracefully
9. **Evaluation Pipeline**: NDCG@10, NDCG@50, MAP, Precision@10 metrics
10. **Comprehensive Documentation**: Architecture, features, benchmarks, usage

## Expected Impact on Metrics

- **NDCG@10**: Improved through better semantic matching, LTR model, and proper tie-breaking
- **NDCG@50**: Enhanced by 70+ expanded features and two-stage retrieval
- **MAP**: Better precision from honeypot detection with statistical anomaly scoring
- **Precision@10**: Higher quality top-10 from semantic retrieval and LTR ranking

## Interview Readiness

All features are explainable and grounded in JD requirements:

- Each feature function includes docstring explaining JD requirement
- Weight rationale documented for interview defensibility
- Reasoning cites concrete facts from profile data
- No black-box components (except pre-computed embeddings)

## Installation

```bash
git clone <YOUR_GITHUB_REPOSITORY>
cd REDROB
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
ource venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Quick Start

**Step 1 — Generate Embeddings (Optional)**
The project automatically generates missing embeddings during the first execution.
To pre-compute embeddings (recommended for faster ranking):

```bash
python embeddings.py
```

This is a one-time preprocessing step.

**Step 2 — Rank Candidates**

```bash
python rank.py --candidates dataset/candidates.jsonl --out submission.csv
```

The pipeline automatically:
Loads the trained LightGBM LambdaMART model (ltr_model.pkl) if available.
Falls back to deterministic manual scoring if the model is unavailable.
Generates the final submission CSV.
Performs internal validation.
Executes the official Redrob validator before completion.

### Reproducibility Checklist

- Clone repository.
- Install dependencies using requirements.txt.
- Generate embeddings (optional, one-time preprocessing).
- Run rank.py.
- Generate submission.csv.
- Automatic internal validation.
- Automatic official Redrob validation.
- Submission ready.
