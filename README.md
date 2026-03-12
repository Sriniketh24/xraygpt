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
git clone https://github.com/yourusername/xraygpt.git
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

## License

MIT

---

*Built as a multimodal AI research project demonstrating Vision Transformer + LLM integration for medical report generation.*
