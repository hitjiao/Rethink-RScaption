# Rethinking Remote Sensing Image Captioning:
From Parameter-Efficient Fine-Tuning to
Knowledge-Driven Training-Free Adapter

## LLM-based Evaluation

This module evaluates the semantic quality of remote sensing image captions by comparing a generated caption (Candidate) with its corresponding ground-truth caption (Reference).

For each Candidate–Reference pair, the evaluation model generates multiple independent judgments using the evaluation prompt. Two criteria are considered:

- **Coverage:** Evaluates whether the Candidate covers the major objects, scene information, and important spatial relationships described in the Reference. Minor details may be omitted as long as the essential semantic content is preserved.
- **Substitutability:** Evaluates whether the Candidate can serve as a valid alternative caption to the Reference by independently conveying the essential scene content and key semantic information without significant factual errors or unsupported claims.

The two criteria are evaluated independently. Linguistic fluency, writing style, and sentence length are not considered unless they affect semantic preservation or understanding.

### 1. Directory Structure

```text
.
├── prompts/
│   └── llm_judgement.txt
├── data/
│   ├── candidate.json
│   └── reference.json
├── results/
│   └── judgement/
├── llm_judgement.py
└── README.md
```

### 2. Requirements

Install the required dependency:

```bash
pip install openai
```

Set your OpenAI API key as an environment variable:

**Linux / macOS**
```bash
export OPENAI_API_KEY="your-api-key"
```

**Windows (PowerShell)**
```powershell
$env:OPENAI_API_KEY="your-api-key"
```

### 3. Input Data Format

The evaluation requires two JSON files:

- `candidate.json`: Generated captions.
- `reference.json`: Corresponding reference captions.

Both files should use the same sample IDs.

Example:

**candidate.json**
```json
{
    "10001": [
        "A residential area with several buildings and surrounding roads."
    ],
    "10002": [
        "An airport with multiple aircraft on the runway."
    ]
}
```

**reference.json**
```json
{
    "10001": [
        "A dense residential area containing several buildings connected by roads."
    ],
    "10002": [
        "An airport featuring multiple aircraft positioned near the runway."
    ]
}
```

Each sample ID corresponds to one Candidate–Reference pair.

### 4. Evaluation

Run the evaluation script:

```bash
python llm_judgement.py \
    --candidate data/candidate.json \
    --reference data/reference.json \
    --prompt_fp prompts/llm_judgement.txt \
    --save_dir results/judgement \
    --model gpt-4o \
    --n 5
```

For Windows:

```bash
python llm_judgement.py ^
    --candidate data/candidate.json ^
    --reference data/reference.json ^
    --prompt_fp prompts/llm_judgement.txt ^
    --save_dir results/judgement ^
    --model gpt-4o ^
    --n 5
```

**Evaluation parameters:**

| Parameter | Description | Default |
|---|---|---|
| `candidate` | Path to the generated captions | Required |
| `reference` | Path to the reference captions | Required |
| `prompt_fp` | Path to the evaluation prompt | Required |
| `save_dir` | Directory for saving evaluation results | `results/judgement` |
| `model` | LLM used for evaluation | `gpt-4o` |
| `n` | Number of independent judgments per sample | `5` |
| `temperature` | Sampling temperature | `2` |
| `retry` | Maximum number of API request attempts | `5` |
| `sleep` | Delay between samples, in seconds | `0.5` |

Here, `n=5` means that the evaluation model generates five judgments for each Candidate–Reference pair. Each judgment independently returns two binary decisions: Coverage and Substitutability.

### 5. Output

The evaluation script generates two files:

| File | Description |
|---|---|
| `judgement_results.json` | Stores the Candidate, Reference, raw model outputs, and parsed judgments for each sample. |
| `judgement_statistics.json` | Stores the aggregate statistics for Coverage and Substitutability. |

Each valid model response follows the format:

```text
{Yes, Yes}
```

The two answers correspond to Coverage and Substitutability, respectively.

### 6. Evaluation Metrics

For each criterion, the Yes Rate is calculated as:

\[
\text{Yes Rate} =
\frac{N_{\text{Yes}}}{N_{\text{Valid}}}
\times 100\%
\]

where:

- \(N_{\text{Yes}}\) is the number of judgments labeled `Yes`.
- \(N_{\text{Valid}}\) is the total number of successfully parsed `Yes` and `No` judgments.

Coverage and Substitutability are calculated independently. Invalid or unparseable outputs are excluded from the corresponding valid judgment count.

A higher Yes Rate indicates that the generated captions more consistently preserve the essential semantic content of the reference captions and satisfy the corresponding evaluation criterion.

### 7. Reproducibility

For reproducible evaluation, we recommend keeping the following settings consistent across experiments:

- LLM model and model version.
- Evaluation prompt.
- Sampling parameters, including `temperature` and `n`.
- Input Candidate and Reference datasets.

The API key should be configured through an environment variable and should never be committed to the repository.




To illustrate this discrepancy, as illustrated in Fig. 1, we project captions from several widely-used natural and RS
datasets into a shared semantic space using MPNet and perform clustering analysis.  The results reveal stark differences:
natural images exhibit dispersed clusters with diverse semantic relationships due to object diversity and varying perspectives.
In contrast, RSIs captured from fixed aerial/satellite perspectives form compact, overlapping clusters dominated by structured geographic entities.

![dc6d1c912fd1e962a945144490c5aa3](https://github.com/user-attachments/assets/9ca5bf0f-4360-49ba-9726-a4cd35d37ccc)

To demonstrate the accuracy and detail of captions generated by our method, we conducted a qualitative comparison with RS-LLaVA, GeoChat, and VHM on two representative scenes: aircraft and ships.
![caption](https://github.com/user-attachments/assets/0ca1fec9-4a24-4606-ae82-488e89e8abac)
![caption_ship](https://github.com/user-attachments/assets/5365ef5c-d901-4513-b0e3-f5ef3d02af08)
