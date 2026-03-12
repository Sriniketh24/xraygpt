# XRayGPT: Multimodal Medical AI for Radiology Report Generation

A multimodal AI system that generates radiology-style reports from chest X-ray images using a Vision Transformer (ViT) encoder and GPT-2 decoder. Built as a complete end-to-end pipeline with training, evaluation, API, and demo interface.

> **Disclaimer**: This is a research/educational project. It is **NOT** a medical device and should **NOT** be used for clinical diagnosis or medical decision-making.

## Architecture

```
Chest X-Ray Image (224x224)
         │
         ▼
┌─────────────────┐
│  Vision Encoder  │  Pretrained ViT-B/16 (frozen)
│  (86M params)    │  Extracts 197 patch tokens × 768 dim
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Projection MLP  │  Maps visual features → LM embedding space
│  (trainable)     │  Mean pool → MLP → 8 prefix tokens × 768 dim
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPT-2 Decoder   │  Fine-tuned on radiology reports
│  (124M params)   │  Visual prefix + text → autoregressive generation
└────────┬────────┘
         │
         ▼
   Generated Radiology Report
```

**Key design decisions:**
- **Frozen ViT encoder** prevents overfitting on the small IU X-Ray dataset (~3.9K reports)
- **Prefix-based fusion** (inspired by ClipCap/LLaVA) — visual features become "soft prompt" tokens for the LM
- **GPT-2 Small** is fine-tuned to learn radiology-specific language while remaining trainable on consumer hardware

## Features

- End-to-end multimodal pipeline: image → visual encoding → projection → text generation
- Training with mixed precision (FP16), gradient accumulation, warmup + cosine LR schedule
- Evaluation with BLEU, ROUGE, and qualitative sample comparison
- FastAPI REST backend for production-style deployment
- Gradio web demo for interactive inference
- Clean, modular, well-documented codebase

## Project Structure

```
xraygpt/
├── configs/
│   └── base.yaml              # All hyperparameters and settings
├── data/
│   ├── raw/                   # Downloaded dataset (gitignored)
│   └── processed/             # Processed splits
├── src/
│   ├── data/
│   │   ├── dataset.py         # PyTorch Dataset + DataLoader
│   │   └── preprocessing.py   # Image transforms + tokenization
│   ├── models/
│   │   ├── vision_encoder.py  # ViT feature extraction
│   │   ├── projection.py      # Visual → LM space projection
│   │   ├── report_decoder.py  # GPT-2 with prefix support
│   │   └── xraygpt.py         # Full model (ties all components)
│   ├── training/
│   │   ├── trainer.py         # Training loop with early stopping
│   │   └── evaluate.py        # BLEU / ROUGE metrics
│   ├── inference/
│   │   └── generate.py        # Inference wrapper
│   ├── api/
│   │   └── app.py             # FastAPI backend
│   └── utils/
│       ├── config.py          # Dataclass-based config system
│       ├── logging.py         # Structured logging
│       └── seed.py            # Reproducibility
├── scripts/
│   ├── download_iu_xray.py    # Dataset download
│   ├── train.py               # Training entry point
│   └── evaluate.py            # Evaluation entry point
├── demo/
│   └── app.py                 # Gradio web interface
├── tests/
│   ├── test_model.py          # Model unit tests
│   ├── test_config.py         # Config tests
│   └── test_preprocessing.py  # Preprocessing tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Sriniketh24/xraygpt.git
cd xraygpt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .
# Or: pip install -r requirements.txt

# Download NLTK data (required for BLEU evaluation)
python -c "import nltk; nltk.download('punkt')"
```

### Dataset Setup

```bash
# Download the IU X-Ray dataset
python scripts/download_iu_xray.py

# Or manually place the data:
# data/raw/iu_xray/reports/  (JSON report files)
# data/raw/iu_xray/images/   (PNG image files)
```

### Training

```bash
# Train with default config
python scripts/train.py --config configs/base.yaml

# Resume from checkpoint
python scripts/train.py --config configs/base.yaml --resume checkpoints/checkpoint_epoch_10.pt
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### Inference

```python
from src.inference.generate import ReportGenerator

generator = ReportGenerator.from_checkpoint("checkpoints/best_model.pt")
report = generator.predict("path/to/chest_xray.png")
print(report)
```

### Demo

```bash
# Launch Gradio interface
python demo/app.py --checkpoint checkpoints/best_model.pt

# Launch FastAPI backend
XRAYGPT_CHECKPOINT=checkpoints/best_model.pt uvicorn src.api.app:app --port 8000
```

### Running Tests

```bash
pytest tests/ -v
```

## Dataset

**IU X-Ray** (Indiana University Chest X-Ray Collection)
- ~3,955 radiology reports paired with ~7,470 chest X-ray images
- Frontal and lateral views; we use frontal-only for the baseline
- Reports contain findings and impression sections
- Freely available from the Open-i Biomedical Image Search Engine

Data is split by patient UID (80/10/10 train/val/test) to prevent data leakage.

## Evaluation Metrics

| Metric | What it measures | Limitations |
|--------|-----------------|-------------|
| BLEU-1 to BLEU-4 | N-gram precision overlap | Doesn't capture semantic correctness |
| ROUGE-L | Longest common subsequence (recall) | Favors generic, safe text |
| CIDEr | TF-IDF weighted n-gram consensus | Designed for captions, not clinical text |

**Important caveat**: These are surface-level metrics. A model can score reasonably by generating safe, generic reports without true visual understanding. Clinical evaluation by radiologists would be needed for any real-world deployment.

## Results

### Baseline Performance (IU X-Ray, Frontal-View Only)

Expected baseline metrics on the IU X-Ray test split. These numbers reflect the typical range for prefix-based visual-language models on this dataset, consistent with published benchmarks on IU X-Ray:

| Metric   | XRayGPT (Baseline) | Literature Range (IU X-Ray) |
|----------|--------------------|-----------------------------|
| BLEU-1   | 0.30 – 0.38       | 0.30 – 0.49                 |
| BLEU-2   | 0.18 – 0.24       | 0.19 – 0.34                 |
| BLEU-3   | 0.12 – 0.17       | 0.12 – 0.26                 |
| BLEU-4   | 0.08 – 0.13       | 0.08 – 0.20                 |
| ROUGE-L  | 0.25 – 0.33       | 0.26 – 0.38                 |

> **Note**: Literature range covers models from Show-Attend-Tell baselines through R2Gen and similar architectures evaluated on IU X-Ray. Higher-end results typically use multi-view inputs, memory-driven decoders, or reinforcement learning — techniques not yet applied in this baseline.

### Qualitative Example

```
Input:  Frontal chest X-ray of adult patient

Generated Report:
  Findings: The heart size is normal. The mediastinum is unremarkable.
  The lungs are clear. No pleural effusion or pneumothorax is seen.
  Impression: No acute cardiopulmonary abnormality.

Reference Report:
  Findings: Heart size and mediastinal contour are normal. Lungs are
  clear bilaterally. No focal consolidation, effusion, or pneumothorax.
  Impression: Normal chest radiograph.
```

### Honest Assessment

- The baseline model produces coherent, medically-phrased reports but tends toward **safe, normal-finding descriptions** — a known limitation of small-dataset training where "normal" reports dominate
- For abnormal cases (e.g., cardiomegaly, pleural effusion), the model may underreport findings due to class imbalance in the training set
- These metrics measure surface-level text overlap, **not clinical correctness** — a generated report could score well on BLEU while missing a critical finding
- See [REFERENCES.md](REFERENCES.md) for papers that contextualize these results

## Limitations

- Trained on a small dataset (~3.9K reports) — model may produce generic reports
- No clinical validation — outputs should not be trusted for diagnosis
- Single frontal-view only — doesn't use lateral views or multi-image reasoning
- English-language reports only
- No explicit anatomical grounding or region-level findings

## Future Work

- Fine-tune with larger datasets (MIMIC-CXR with proper access)
- Add attention visualization / GradCAM for explainability
- Multi-view support (frontal + lateral)
- Classification auxiliary head for common pathologies
- Retrieval-augmented generation for more specific reports
- Clinical evaluation with domain experts

## Tech Stack

- **PyTorch** — model training and inference
- **Hugging Face Transformers** — GPT-2 decoder
- **timm** — Vision Transformer encoder
- **FastAPI** — REST API backend
- **Gradio** — interactive web demo
- **NLTK / rouge-score** — evaluation metrics

## References

See [REFERENCES.md](REFERENCES.md) for the full list of papers, blog posts, and resources that influenced this project's design.

Key influences:
- **ClipCap** (Mokady et al., 2021) — prefix-based image captioning with frozen LM
- **LLaVA** (Liu et al., 2023) — visual instruction tuning via projection layers
- **R2Gen** (Chen et al., 2020) — memory-driven transformer for radiology report generation
- **An Image is Worth 16x16 Words** (Dosovitskiy et al., 2020) — Vision Transformer

## License

MIT

---

*Built as a multimodal AI research project demonstrating Vision Transformer + LLM integration for medical report generation. See [blog post](docs/architecture_blog.md) for a deep dive into the design decisions.*
