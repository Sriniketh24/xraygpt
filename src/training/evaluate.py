"""
Evaluation metrics for radiology report generation.

Implements:
  - BLEU (1-4): Measures n-gram precision. Standard MT metric. Useful but
    can miss semantic meaning — a report can be factually correct with low BLEU.

  - ROUGE-L: Measures longest common subsequence. Captures recall of
    reference content. More forgiving than BLEU for paraphrased content.

  - CIDEr: Consensus-based metric using TF-IDF weighted n-grams. Designed
    for image captioning. Rewards specificity over generic descriptions.

Limitations:
  - All metrics are surface-level (n-gram overlap). They don't measure
    clinical accuracy or factual correctness.
  - A model generating safe, generic reports may score reasonably on these
    metrics without being clinically useful.
  - For a production system, domain expert evaluation would be essential.
"""

from collections import defaultdict

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer


def compute_bleu(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    """
    Compute corpus-level BLEU-1 through BLEU-4.

    Args:
        references: list of reference report strings
        hypotheses: list of generated report strings

    Returns:
        dict with bleu_1, bleu_2, bleu_3, bleu_4
    """
    # Tokenize
    refs_tokenized = [[ref.lower().split()] for ref in references]
    hyps_tokenized = [hyp.lower().split() for hyp in hypotheses]

    smooth = SmoothingFunction().method1

    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        score = corpus_bleu(
            refs_tokenized,
            hyps_tokenized,
            weights=weights,
            smoothing_function=smooth,
        )
        scores[f"bleu_{n}"] = score

    return scores


def compute_rouge(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        references: list of reference report strings
        hypotheses: list of generated report strings

    Returns:
        dict with rouge_1, rouge_2, rouge_l (F1 scores)
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    aggregated = defaultdict(list)

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for key, value in scores.items():
            aggregated[key].append(value.fmeasure)

    return {
        "rouge_1": np.mean(aggregated["rouge1"]),
        "rouge_2": np.mean(aggregated["rouge2"]),
        "rouge_l": np.mean(aggregated["rougeL"]),
    }


def compute_all_metrics(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        references: list of reference report strings
        hypotheses: list of generated report strings

    Returns:
        dict with all metric scores
    """
    metrics = {}
    metrics.update(compute_bleu(references, hypotheses))
    metrics.update(compute_rouge(references, hypotheses))
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics as a readable string."""
    lines = ["Evaluation Metrics", "=" * 40]
    for key, value in sorted(metrics.items()):
        lines.append(f"  {key:<12s}: {value:.4f}")
    lines.append("=" * 40)
    return "\n".join(lines)
