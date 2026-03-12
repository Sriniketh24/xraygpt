# Building XRayGPT: Architecture Decisions for Multimodal Medical Report Generation

*A deep dive into why each technical decision was made — what worked, what the tradeoffs are, and what I'd do differently at scale.*

---

## The Problem

Radiologists examine chest X-rays and write structured reports describing what they see: heart size, lung clarity, presence of abnormalities, and an overall impression. This is a **visually-grounded language generation** task — the model must look at an image and produce coherent, factually relevant text.

This is harder than standard image captioning for three reasons:
1. **Domain specificity**: medical language has precise terminology and structure
2. **Clinical stakes**: wrong outputs could mislead (though this project is strictly educational)
3. **Long-form output**: reports are multi-sentence, not single captions

The goal was to build the strongest realistic system I could as a student, using publicly available data and consumer hardware.

---

## Architecture Overview

```
Chest X-Ray → ViT-B/16 → Projection MLP → GPT-2 → Report
               (frozen)     (trained)       (fine-tuned)
```

Three components, each with a clear role. Let me walk through why each one was chosen.

---

## Decision 1: Vision Encoder — Why ViT over CNNs?

**Choice**: ViT-B/16 pretrained on ImageNet-21k, frozen during training.

**Alternatives considered**:
- DenseNet-121 pretrained on CheXpert (medical-domain CNN)
- ResNet-50 (standard baseline)
- BiomedCLIP (medical vision-language model)

**Why ViT won**:

ViT naturally produces a **sequence of patch embeddings** (197 tokens for a 224x224 image with 16x16 patches). This is exactly what we need — a sequence of visual tokens that can be projected into the language model's embedding space. With a CNN, we'd need to flatten feature maps or add a custom pooling step to create a token sequence.

The patch-level representation also preserves spatial locality. Patch token 42 roughly corresponds to a specific region of the X-ray. This matters for radiology because findings are spatially grounded ("right lower lobe opacity").

**Why frozen**: The IU X-Ray dataset has only ~3,900 reports. Fine-tuning an 86M-parameter ViT on this would overfit badly. Freezing the encoder lets us use its strong pretrained representations while only training the smaller downstream components. This is a standard transfer learning strategy, validated by both ClipCap and LLaVA.

**What I'd change at scale**: With a larger dataset like MIMIC-CXR (377K images), I'd consider either (a) unfreezing the encoder for end-to-end fine-tuning, or (b) swapping in a medical-domain pretrained encoder like BiomedCLIP to get domain-specific visual features.

---

## Decision 2: Projection Layer — The Bridge Between Vision and Language

**Choice**: 2-layer MLP that mean-pools ViT patch tokens, then projects to 8 "visual prefix" tokens in GPT-2's 768-dim embedding space.

This is the most important architectural component because it's where the cross-modal alignment happens.

**How it works**:

```
ViT output: (batch, 197, 768)     # 197 patch tokens
    → Mean pool: (batch, 768)      # single visual vector
    → MLP: (batch, 8 × 768)        # project to prefix size
    → Reshape: (batch, 8, 768)     # 8 visual prefix tokens
    → LayerNorm                     # stabilize training
```

**Why 8 prefix tokens?** This is a hyperparameter balancing expressiveness vs. computational cost. More prefix tokens give the decoder more "visual context" to attend to, but also consume more of GPT-2's 1024-token context window. 8 tokens is a common starting point in the literature (ClipCap uses 10, LLaVA uses variable-length). With max report length of 128, we use 8 + 128 = 136 of the 1024 available positions.

**Why mean pooling first?** Mean pooling compresses the spatial information into a single vector before the MLP expands it into prefix tokens. This forces the MLP to learn a compact visual representation. An alternative would be to keep all 197 patch tokens and project each one individually, but this would make the prefix 197 tokens long, consuming most of GPT-2's context window and significantly increasing compute.

**The MLP itself**: Two linear layers with GELU activation, dropout, and a final LayerNorm. GELU was chosen over ReLU because transformers (both ViT and GPT-2) use GELU internally, so the projection layer operates in a compatible activation space.

---

## Decision 3: Text Decoder — Why GPT-2 Small?

**Choice**: GPT-2 Small (124M parameters), fine-tuned on radiology reports.

**Alternatives considered**:
- GPT-2 Medium (355M) — better capacity but harder to fit in memory
- BioGPT — medical-domain GPT, but less mature tooling
- LSTM decoder — simpler, but weaker at capturing long-range dependencies
- Larger LLMs (LLaMA, etc.) — impractical for full fine-tuning on one GPU

**Why GPT-2 Small**:
1. **124M params fits in consumer GPU memory** alongside the frozen ViT (~86M params)
2. **Strong pretrained language model** — GPT-2 generates fluent English, so we only need to teach it medical vocabulary and report structure
3. **Excellent HuggingFace support** — well-tested generation code, tokenizer, and KV caching
4. **Fast iteration** — training completes in hours, not days

**Why not freeze GPT-2 (like ClipCap)?** ClipCap showed you can generate decent captions with a frozen GPT-2. But radiology reports use specialized vocabulary ("cardiomegaly", "pneumothorax", "costophrenic angle") that GPT-2 rarely saw during pretraining. Fine-tuning the decoder is essential for domain adaptation.

**The prefix mechanism**: During training, the input to GPT-2 is `[visual_prefix_tokens | report_tokens]`. The loss is computed only on the report tokens (prefix positions are masked with -100 in the labels). This means the model learns to generate text *conditioned on* the visual prefix — the prefix acts like a "soft prompt" that tells GPT-2 what's in the image.

During inference, we feed only the visual prefix and let GPT-2 autoregressively generate the report token by token.

---

## Decision 4: Training Strategy

**Approach**: Freeze ViT, train projection + GPT-2 with AdamW, linear warmup → cosine decay.

Key training details:
- **FP16 mixed precision**: Halves memory usage with negligible accuracy impact
- **Gradient accumulation (2 steps)**: Effective batch size of 32 with batch size 16
- **Early stopping (patience=7)**: Prevents overfitting on the small dataset
- **Gradient clipping (max_norm=1.0)**: Prevents exploding gradients during LM fine-tuning

**Learning rate**: 5e-5 is the standard starting point for transformer fine-tuning (established by BERT, validated across hundreds of papers). With the cosine schedule, the LR smoothly decays to near-zero by the end of training, which helps with final convergence.

**What to watch for during training**:
- **Validation loss plateaus early**: Expected with small datasets. The model quickly learns the "normal report" template and then struggles to improve on abnormal cases
- **Generated reports become repetitive**: A sign of mode collapse. Repetition penalty (1.2) during generation helps, but the root cause is limited training data diversity

---

## Decision 5: Dataset — Why IU X-Ray?

**Choice**: Indiana University Chest X-Ray Collection (~3,955 reports, ~7,470 images).

| Dataset | Size | Access | Practicality |
|---------|------|--------|-------------|
| IU X-Ray | ~3.9K reports | Free, open access | High |
| MIMIC-CXR | ~377K reports | PhysioNet credentialing required | Low for quick projects |
| CheXpert | ~224K images | Free, but classification-only (no reports) | N/A for report generation |
| PadChest | ~160K images | Free, Spanish reports | Language barrier |

IU X-Ray is the standard benchmark for medical report generation research. It's small but well-studied — R2Gen, M2Transformer, and many other papers report results on it. This makes it easy to contextualize our results.

**Data split strategy**: We split by patient UID, not by image. This prevents a subtle form of data leakage where training and test sets contain different images from the same patient (which would make the task artificially easier, since reports for the same patient are often similar).

---

## Decision 6: Evaluation — Being Honest About Metrics

We evaluate with BLEU-1 through BLEU-4 and ROUGE-L. These are standard in the field and enable comparison with published work.

But I want to be explicit about what these metrics **don't** measure:

1. **Clinical accuracy**: A report saying "no pneumothorax" when there IS a pneumothorax would still score well on BLEU if the reference also discusses pneumothorax
2. **Factual grounding**: The model could generate a fluent report that describes findings not present in the image
3. **Completeness**: Missing a critical finding has no special penalty beyond lower n-gram overlap

For a production system, you'd need:
- Radiologist evaluation (precision/recall on specific findings)
- CheXpert labeler or similar tools to extract and compare structured findings
- Error analysis stratified by pathology type

For a research project, the standard metrics + honest documentation of limitations is the right approach.

---

## What I'd Do Differently With More Resources

1. **Larger dataset**: MIMIC-CXR (377K images) would dramatically reduce the generic-report problem
2. **Medical vision encoder**: BiomedCLIP or a CheXpert-pretrained ViT would provide domain-specific visual features
3. **Larger decoder**: GPT-2 Medium or a medical LLM like BioGPT for richer generation
4. **Multi-view fusion**: Use both frontal and lateral views (cross-attention between view embeddings)
5. **Reinforcement learning**: Optimize directly for CIDEr or a clinical metric rather than cross-entropy
6. **Retrieval augmentation**: Retrieve similar training reports to condition generation, reducing hallucination

---

## Key Takeaways

1. **Prefix-based fusion is elegant and practical** — a simple MLP projecting visual features into LM space works surprisingly well and is easy to train
2. **Freezing the vision encoder is essential for small datasets** — without this, the model memorizes training images
3. **GPT-2 Small is a sweet spot** — powerful enough for coherent text, small enough for single-GPU training
4. **Honest evaluation matters more than high numbers** — understanding why BLEU=0.35 is both reasonable and insufficient teaches more than chasing state-of-the-art
5. **The hardest part isn't the architecture, it's the data** — with 3,900 reports, even a perfect architecture will produce somewhat generic outputs

---

*This post documents the design decisions behind [XRayGPT](https://github.com/Sriniketh24/xraygpt). The project is for research and educational purposes only.*
