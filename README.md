# LLM Classification Finetuning — Kaggle Competition

A collection of notebooks for the [Kaggle LLM Classification Finetuning competition](https://www.kaggle.com/competitions/llm-classification-finetuning), which challenges participants to predict which of two LLM responses a human evaluator preferred — or whether they tied.

## Problem Statement

Given a user **prompt** and two LLM-generated **responses** (A and B), predict the winner:

| Label | Meaning |
|---|---|
| `winner_model_a` | Response A was preferred |
| `winner_model_b` | Response B was preferred |
| `winner_tie` | Neither was clearly better |

This is a 3-class classification problem evaluated on **log loss**.

---

## Repository Structure

```
LLM-Classification-Finetuning-Kaggle-main/
│
├── llm_classification_logistic_regression_TFID.ipynb   # Baseline: TF-IDF + Logistic Regression
├── llm_classification_bert.ipynb                       # DeBERTa-v3-base fine-tuning
├── llm_classification_llama_lora.ipynb                 # Llama 3.2-1B + LoRA (4-bit QLoRA)
└── qwen-lora-v1.ipynb                                  # Qwen2.5-0.5B + LoRA (4-bit QLoRA)
```

---

## Approaches

### 1. Baseline — TF-IDF + Logistic Regression (`llm_classification_logistic_regression_TFID.ipynb`)

A classical NLP baseline for quick iteration and benchmarking.

- Text preprocessing: lowercasing, punctuation removal, stopword removal, lemmatization (NLTK)
- Feature extraction: TF-IDF vectorization over concatenated prompt + responses
- EDA included: duplicate checking, null analysis, LLM identity exploration
- Good for establishing a floor score without GPU requirements

### 2. DeBERTa Fine-tuning (`llm_classification_bert.ipynb`)

A BERT-family approach using Microsoft's DeBERTa-v3-base, one of the strongest encoder models for classification.

**Key design choices:**
- Input format: `[CLS] prompt [SEP] response_a [SEP] response_b` concatenated and truncated to 1024 tokens
- **Swap augmentation**: A/B responses are randomly swapped with label inversion (effectively doubles training data and reduces position bias)
- 3-fold stratified cross-validation
- Mixed precision training (FP16) with gradient accumulation for an effective batch size of 32
- Multi-GPU support via `DataParallel`
- AdamW optimizer with cosine LR schedule and warmup

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Model | `microsoft/deberta-v3-base` |
| Max sequence length | 1024 |
| Epochs | 2 |
| Batch size (per GPU) | 4 |
| Gradient accumulation | 4 |
| Learning rate | 2e-5 |
| Folds | 3 |
| Hardware | Kaggle T4 ×2 |

### 3. Llama 3.2-1B + LoRA (`llm_classification_llama_lora.ipynb`)

A parameter-efficient fine-tuning approach using a small Llama model with QLoRA (4-bit quantization + LoRA adapters).

**Key design choices:**
- Smart truncation: keeps the beginning and end of each text, drops the repetitive middle — preserves structure and conclusions within the token budget
- Markdown stripping: removes fences, bold/italic markers, and headers to maximize semantic signal per token
- 4-bit quantization (`BitsAndBytesConfig`) for memory efficiency

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-1B` |
| Max sequence length | 512 |
| Batch size | 32 |
| Learning rate | 2e-4 |
| Epochs | 1 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |

### 4. Qwen2.5-0.5B + LoRA (`qwen-lora-v1.ipynb`)

An alternative small-LLM approach using Alibaba's Qwen2.5-0.5B with the same QLoRA recipe as the Llama notebook. Qwen2.5-0.5B is notably compact, making it fast to train on a single GPU.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Max sequence length | 512 |
| Batch size | 32 |
| Learning rate | 2e-4 |
| Epochs | 1 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |

---

## Key Techniques

**Swap Augmentation** (DeBERTa notebook): Since the model must be position-agnostic (it shouldn't always favor response A just because it comes first), every training sample is duplicated with A and B swapped and the label inverted. This doubles the effective training set and meaningfully reduces position bias.

**Smart Truncation** (LoRA notebooks): Rather than naively truncating from the right, the smart truncation strategy keeps the first half and last half of each field, dropping the middle. This preserves the opening (which establishes structure and context) and the conclusion (which often contains the key decision signal).

**4-bit QLoRA**: The LoRA notebooks use `BitsAndBytesConfig` to load the base model in 4-bit precision, drastically reducing VRAM usage. LoRA adapters (a small fraction of total parameters) are trained in full precision on top.

---

## Setup & Requirements

```bash
# BERT / DeBERTa notebook
pip install transformers==4.40.1 datasets accelerate sentencepiece protobuf

# LoRA notebooks (Llama & Qwen)
pip install bitsandbytes>=0.46.1 transformers peft accelerate datasets
```

All notebooks are designed to run on **Kaggle** (free T4 GPU tier). The DeBERTa notebook takes advantage of dual-GPU setups (T4 ×2). The LoRA notebooks run on a single GPU.

Data is loaded from the Kaggle competition directly:
```python
import kagglehub
kagglehub.competition_download('llm-classification-finetuning')
```

---

## Data Format

| Column | Description |
|---|---|
| `id` | Unique row identifier |
| `prompt` | The user prompt shown to both models |
| `response_a` | Response from model A |
| `response_b` | Response from model B |
| `model_a` | Identity of model A (training only) |
| `model_b` | Identity of model B (training only) |
| `winner_model_a` | 1 if A won |
| `winner_model_b` | 1 if B won |
| `winner_tie` | 1 if tied |

---

## Approach Comparison

| Notebook | Model | GPU Required | Strengths |
|---|---|---|---|
| TF-IDF + LR | — | No | Fast baseline, interpretable |
| DeBERTa | deberta-v3-base | Yes (T4 ×2) | Strong encoder, swap augmentation, cross-val |
| Llama LoRA | Llama-3.2-1B | Yes (T4 ×1) | Generative model, QLoRA efficient |
| Qwen LoRA | Qwen2.5-0.5B | Yes (T4 ×1) | Smallest model, fastest to train |

---

## License

MIT — see `LICENSE` for details.