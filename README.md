> **Rethinking Remote Sensing Image Captioning: From Parameter-Efficient Fine-Tuning to Domain Knowledge-Driven Alignment-Free Framework**  

## Repository Structure
```
Rethink-RScaption/
├── Visual Prior Extraction/          # Stage 1 + Stage 2: perception models as HTTP services
│   ├── class_inf.py                  #   scene classification service  (port 1203)
│   ├── def_inf.py                    #   object detection service     (port 1212)
│   ├── seg_inf.py                    #   semantic segmentation service (port 1214)
│   ├── yolo_plane.py                 #   fine-grained aircraft classifier (23 classes)
│   └── yolo_ship.py                  #   fine-grained ship classifier     (8 classes)
│
├── Scene Knowledge Base Prompt/      # Stage 2: prompt used to build the 45-scene rule base
│   └── Scene Knowledge Base Prompt.txt
│
├── LLM-based Summary/                # Stage 3: CoT prompt for caption synthesis
│   └── LLM-based summary.txt
│
└── LLM-based Evaluation/             # Evaluation: LLM-as-a-Ranker / LLM-as-a-Judge
    ├── LLM_ranking.py
    ├── LLM_judgement.py
    ├── accuracy_rank_prompt.txt
    ├── fineness_rank_prompt.txt
    ├── fluency_rank_prompt.txt
    ├── richness_rank_prompt.txt
    └── LLM-judge.txt
```
## Installation

```bash
conda create -n rscaption python=3.8 -y
conda activate rscaption

# PyTorch (CUDA 11.7 as an example; any version compatible with your MMCV build works)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html

# OpenMMLab stack (CLS / DET / SEG inference)
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" mmpretrain mmdet mmsegmentation

# Runtime dependencies
pip install flask openai ultralytics pillow opencv-python numpy requests
```

> **Note**  
> The three perception services are thin wrappers around the [MTP](https://github.com/ViTAE-Transformer/MTP) toolbox. Please use the **same MMCV / MMDetection / MMSegmentation versions** as the MTP repository, otherwise its config files may fail to build.
---

## Quick Start
The whole framework is a **training-free pipeline**: three local HTTP services + one LLM call.

```bash
# 0) Put the downloaded checkpoints / configs in place (see "Model Weights")
#    ./configs/mtp/...                    <- copied from the MTP repo
#    ./resisc45-rvsa-l-mae-mtp-epoch_200.pth
#    ./dior.pth
#    ./loveda-rvsa-l-mae-mtp-iter_80000.pth

# 1) Launch the three perception services (three terminals, same conda env)
cd "Visual Prior Extraction"
python class_inf.py     # 0.0.0.0:1203  -> /class_english
python def_inf.py       # 0.0.0.0:1212  -> /det_english
python seg_inf.py       # 0.0.0.0:1214  -> /seg_english

# 2) Query them with raw image bytes
curl -X POST --data-binary @"test.jpg" http://127.0.0.1:1203/class_english
curl -X POST --data-binary @"test.jpg" http://127.0.0.1:1212/det_english
curl -X POST --data-binary @"test.jpg" http://127.0.0.1:1214/seg_english

# 3) Assemble the textual representation + retrieved knowledge, then call the LLM
python your_pipeline.py --image test.jpg
```
The three text blocks plus the retrieved scene context are inserted into the CoT prompt of [`LLM-based Summary/LLM-based summary.txt`](LLM-based%20Summary/LLM-based%20summary.txt) to obtain the caption.


## Module · LLM-based Evaluation

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
python LLM_ranking.py \
    --candidate1 data/candidate1.json \
    --candidate2 data/candidate2.json \
    --candidate3 data/candidate3.json \
    --candidate4 data/candidate4.json \
    --candidate5 data/candidate5.json \
    --reference data/reference.json \
    --accuracy_prompt accuracy_rank_prompt.txt \
    --fineness_prompt fineness_rank_prompt.txt \
    --fluency_prompt fluency_rank_prompt.txt \
    --richness_prompt richness_rank_prompt.txt \
    --save_dir results/ranking \
    --model gpt-4o \
    --n 5
```

### 2. LLM-as-a-Judge Evaluation

This method evaluates each Candidate against its corresponding Reference using two independent criteria:

- **Coverage:** Whether the Candidate preserves the major objects, scene information, and important spatial relationships in the Reference.
- **Substitutability:** Whether the Candidate can serve as a valid alternative caption while preserving the essential semantic content of the Reference.

For each Candidate–Reference pair, the LLM generates five independent judgments (`n=5`). Each judgment returns two binary decisions (`Yes` or `No`), corresponding to Coverage and Substitutability.

**Run:**

```bash
python LLM_judgement.py \
    --candidate data/candidate.json \
    --reference data/reference.json \
    --prompt_fp LLM-judge.txt \
    --save_dir results/judgement \
    --model gpt-4o \
    --n 5
```


## Datasets

### 1. Fine-grained Aircraft & Ship Dataset

**Aircraft (23 classes)**

| Group                   | Classes                                               |
| ----------------------- | ----------------------------------------------------- |
| Civil Airliner          | ARJ21, Boeing 747, Boeing 777                         |
| Bomber                  | B-1B, B-52, TU-22, TU-95, TU-160, SU-24, SU-34, SU-35 |
| Fighter Jet             | F-15, F-16, F-22, FA-18                               |
| Transport Aircraft      | C-5, C-17, C-130                                      |
| Stratotanker            | KC-10, KC-135                                         |
| AWACS                   | E-3, E-8                                              |
| Anti-submarine Aircraft | P-3C                                                  |

**Ship (8 classes)**

| Group         | Classes                                                                       |
| ------------- | ----------------------------------------------------------------------------- |
| Warship       | Aircraft Carrier, Amphibious Assault Ship, Amphibious Transport Dock, Cruiser |
| Civilian Ship | Dry Cargo Ship, Liquid Cargo Ship, Motorboat, Tugboat                         |


### 2. NWPU-RESISC45 (R45)
- Official page: <https://gcheng-nwpu.github.io/#datasets>
- OneDrive: <https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs>
- Baidu Netdisk: <https://pan.baidu.com/s/1mifR6tU>
- HuggingFace mirror: <https://huggingface.co/datasets/timm/resisc45>

### 3. DIOR
- Official page: <https://gcheng-nwpu.github.io/#datasets>
- Google Drive: <https://drive.google.com/open?id=1UdlgHk49iu6WpcJ5467iT-UqNPpx\_\_CC>
- Baidu Netdisk: <https://pan.baidu.com/s/1iLKT0JQoKXEJTGNxt5lSMg>
- Rotated version (DIOR-R, not used here): <https://gcheng-nwpu.github.io/#datasets>

### 4. LoveDA
- Official repository: <https://github.com/Junjue-Wang/LoveDA>
- Zenodo: <https://doi.org/10.5281/zenodo.5706578>
- Baidu Netdisk: <https://pan.baidu.com/s/1YrU1Y4Y0dS0f_OOHXpzspQ> (code: `27vc`)

### 5. Captioning benchmarks used for evaluation
- **Sydney-Captions** — 613 images × 5 human captions.
- **UCM-Captions** — 2,100 aerial images, 21 scene classes, 5 captions each.
- **RSICD** — 10,921 images, 30 scene classes; we split it into ID / OOD subsets for the generalization study.

