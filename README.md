# Rethinking Remote Sensing Image Captioning:
From Parameter-Efficient Fine-Tuning to Knowledge-Driven Training-Free Adapter

## LLM-based Evaluation

We provide two LLM-based evaluation methods to assess the quality of remote sensing image captions.

### 1. Ranking-based Evaluation

This method compares five candidate captions and ranks them according to four evaluation dimensions:

- **Accuracy:** Correctness of scene categories, object categories, object quantities, spatial relationships, and scene-related information.
- **Fineness:** Specificity of object locations, spatial relationships, and fine-grained object categories.
- **Fluency:** Sentence structure, grammatical correctness, and overall coherence.
- **Richness:** Overall scene coverage and the diversity of visual details.

For each dimension, the LLM ranks the five candidates and outputs a five-digit sequence, where each digit represents the rank of the corresponding candidate (1 = best, 5 = worst).

**Run:**

```bash
python evaluate_all.py \
    --candidate1 data/candidate1.json \
    --candidate2 data/candidate2.json \
    --candidate3 data/candidate3.json \
    --candidate4 data/candidate4.json \
    --candidate5 data/candidate5.json \
    --reference data/reference.json \
    --prompt_dir prompts/ \
    --save_dir results/ranking \
    --n 5
```

### 2. LLM-as-a-Judge Evaluation

This method evaluates each Candidate against its corresponding Reference using two independent criteria:

- **Coverage:** Whether the Candidate preserves the major objects, scene information, and important spatial relationships in the Reference.
- **Substitutability:** Whether the Candidate can serve as a valid alternative caption while preserving the essential semantic content of the Reference.

For each Candidate–Reference pair, the LLM generates five independent judgments (`n=5`). Each judgment returns two binary decisions (`Yes` or `No`), corresponding to Coverage and Substitutability.

**Run:**

```bash
python llm_judgement.py \
    --candidate data/candidate.json \
    --reference data/reference.json \
    --prompt_fp prompts/llm_judgement.txt \
    --save_dir results/judgement \
    --model gpt-4o \
    --n 5
```

