# References & Influential Works

Papers, blog posts, and resources that directly influenced XRayGPT's architecture, training strategy, and evaluation approach.

---

## Core Architecture Influences

### Vision Transformer (ViT)

- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
  Dosovitskiy et al., ICLR 2021
  [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

  *Why it matters*: Introduced the Vision Transformer, showing that a pure transformer applied to sequences of image patches can match or exceed CNNs on image classification. XRayGPT uses ViT-B/16 as its visual backbone. The key insight — treating an image as a sequence of patch embeddings — directly enables our prefix-based fusion with the language model.

### Prefix-Based Visual Conditioning

- **ClipCap: CLIP Prefix for Image Captioning**
  Mokady et al., 2021
  [arXiv:2111.09734](https://arxiv.org/abs/2111.09734)

  *Why it matters*: Demonstrated that a simple MLP projecting CLIP image embeddings into a "prefix" for GPT-2 can generate strong image captions — without fine-tuning the language model at all. XRayGPT's projection layer is directly inspired by this approach. We chose to also fine-tune GPT-2 (unlike ClipCap's frozen-LM variant) because radiology language is domain-specific and unlikely to be well-represented in GPT-2's pretraining data.

### Visual Instruction Tuning

- **Visual Instruction Tuning (LLaVA)**
  Liu et al., NeurIPS 2023
  [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)

  *Why it matters*: Showed that a linear projection from a vision encoder to an LLM's embedding space, combined with instruction tuning, produces strong multimodal reasoning. LLaVA validated the "vision encoder + projection + LM" paradigm at scale. XRayGPT uses the same structural pattern but at a much smaller scale (ViT-B + GPT-2 vs. CLIP-ViT-L + Vicuna-13B), making it tractable for single-GPU training.

---

## Medical Report Generation

### Memory-Driven Transformers

- **Generating Radiology Reports via Memory-driven Transformer (R2Gen)**
  Chen et al., EMNLP 2020
  [arXiv:2010.16056](https://arxiv.org/abs/2010.16056)

  *Why it matters*: Introduced a relational memory module to help the decoder maintain consistency across long radiology reports. R2Gen is one of the strongest baselines on the IU X-Ray dataset. Our baseline doesn't include a memory mechanism (future work), but R2Gen's results on IU X-Ray provide the primary benchmark context for our evaluation table.

### Show, Attend and Tell

- **Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**
  Xu et al., ICML 2015
  [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)

  *Why it matters*: The foundational attention-based image captioning model. While XRayGPT uses a different architecture (prefix-based rather than attention-at-each-step), this paper established the encoder-decoder paradigm for image-to-text that all subsequent work builds on. Understanding this lineage is essential for explaining why we chose a prefix approach over classical attention.

### Automated Radiology Reporting

- **On the Automatic Generation of Medical Imaging Reports**
  Jing et al., ACL 2018
  [arXiv:1711.08195](https://arxiv.org/abs/1711.08195)

  *Why it matters*: One of the first papers to specifically tackle radiology report generation (as opposed to general image captioning). Introduced the co-attention mechanism and hierarchical LSTM for generating multi-sentence reports. Established IU X-Ray as a standard benchmark for this task.

- **Knowledge Matters: Radiology Report Generation with General and Specific Knowledge**
  Yang et al., Medical Image Analysis, 2022
  [arXiv:2112.15009](https://arxiv.org/abs/2112.15009)

  *Why it matters*: Showed that incorporating medical knowledge graphs improves report specificity — addressing the "generic report" problem that affects all small-dataset models, including our baseline.

---

## Training & Optimization

### GPT-2 and Autoregressive Language Modeling

- **Language Models are Unsupervised Multitask Learners (GPT-2)**
  Radford et al., OpenAI 2019
  [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

  *Why it matters*: GPT-2 is our text decoder. Its byte-pair encoding tokenizer and autoregressive architecture form the generation backbone. We chose GPT-2 Small (124M) specifically because it's large enough to produce fluent text but small enough to fine-tune on a single GPU with the IU X-Ray dataset.

### Mixed Precision Training

- **Mixed Precision Training**
  Micikevicius et al., ICLR 2018
  [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

  *Why it matters*: FP16 training reduces memory usage and speeds up training with minimal accuracy loss. Essential for fitting our model (ViT-B + GPT-2 = ~210M params) into consumer GPU memory during training.

### Learning Rate Scheduling

- **SGDR: Stochastic Gradient Descent with Warm Restarts**
  Loshchilov & Hutter, ICLR 2017
  [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)

  *Why it matters*: Introduced cosine annealing schedules. Our training pipeline uses linear warmup followed by cosine decay, which is the standard practice for transformer fine-tuning and helps with training stability.

---

## Evaluation

### BLEU

- **BLEU: a Method for Automatic Evaluation of Machine Translation**
  Papineni et al., ACL 2002
  [Paper](https://aclanthology.org/P02-1040/)

  *Why it matters*: The most widely used metric for text generation. We compute BLEU-1 through BLEU-4. Important limitation: BLEU measures n-gram precision, not semantic correctness. A report that is factually wrong but uses similar words to the reference will still score well.

### ROUGE

- **ROUGE: A Package for Automatic Evaluation of Summaries**
  Lin, ACL Workshop 2004
  [Paper](https://aclanthology.org/W04-1013/)

  *Why it matters*: Complements BLEU by measuring recall (how much of the reference is captured). ROUGE-L uses longest common subsequence, which is more forgiving of word order variations than BLEU.

### CIDEr

- **CIDEr: Consensus-based Image Description Evaluation**
  Vedantam et al., CVPR 2015
  [arXiv:1411.5726](https://arxiv.org/abs/1411.5726)

  *Why it matters*: Uses TF-IDF weighting to reward descriptions that are specific to an image rather than generic. More informative than BLEU for image captioning tasks. However, it was designed for short captions, not multi-sentence radiology reports, so its utility is limited in our context.

---

## Datasets

### IU X-Ray

- **Preparing a Collection of Radiology Examinations for Distribution and Retrieval**
  Demner-Fushman et al., JAMIA 2016
  [Paper](https://pubmed.ncbi.nlm.nih.gov/26133894/)

  *Why it matters*: The IU X-Ray dataset paper. Describes the collection of ~3,955 reports and ~7,470 chest X-ray images from Indiana University hospitals. This is our primary training and evaluation dataset, chosen for its accessibility and established use as a benchmark.

### MIMIC-CXR (Context)

- **MIMIC-CXR, a De-identified Publicly Available Database of Chest Radiographs with Free-text Reports**
  Johnson et al., Scientific Data 2019
  [Paper](https://physionet.org/content/mimic-cxr/2.0.0/)

  *Why it matters*: The largest publicly available chest X-ray dataset (377K images). We don't use it in the baseline due to access requirements (PhysioNet credentialing) and computational cost, but it represents the natural next step for scaling up training.

---

## Blog Posts & Practical Resources

- **The Illustrated Transformer** — Jay Alammar
  [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
  Clear visual explanation of the transformer architecture underlying both ViT and GPT-2.

- **Vision Transformers (ViT) in Image Recognition** — Google AI Blog
  [ai.googleblog.com](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html)
  Practical overview of ViT design choices and scaling behavior.

- **Hugging Face Transformers Documentation**
  [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
  Reference for GPT-2 model usage, tokenization, and generation APIs.

- **timm Documentation**
  [huggingface.co/docs/timm](https://huggingface.co/docs/timm)
  Reference for ViT model loading and configuration.

---

## How These Influenced XRayGPT

| Design Decision | Primary Influence |
|----------------|-------------------|
| ViT-B/16 as visual encoder | Dosovitskiy et al. (ViT) |
| Projection MLP to prefix tokens | Mokady et al. (ClipCap), Liu et al. (LLaVA) |
| GPT-2 Small as decoder | Radford et al. (GPT-2) |
| Freeze encoder, train projection + decoder | ClipCap (frozen approach), LLaVA (projection training) |
| IU X-Ray dataset | Demner-Fushman et al., standard in R2Gen and related work |
| BLEU/ROUGE evaluation + honesty about limits | Papineni et al., Lin, community critique of n-gram metrics |
| Cosine LR with warmup | Loshchilov & Hutter (SGDR), standard transformer practice |
| FP16 mixed precision | Micikevicius et al. |
