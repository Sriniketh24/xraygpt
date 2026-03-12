"""
Evaluation script for XRayGPT.

Generates reports for the test set and computes metrics.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --config configs/base.yaml \
        --output outputs/evaluation_results.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.inference.generate import ReportGenerator
from src.training.evaluate import compute_all_metrics, format_metrics
from src.utils.config import load_config
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate XRayGPT")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Config file override"
    )
    parser.add_argument(
        "--test_csv", type=str, default=None, help="Path to test split CSV"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Limit number of samples to evaluate (for quick testing)"
    )
    args = parser.parse_args()

    logger = setup_logger("xraygpt.eval")

    # Load generator
    logger.info("Loading model...")
    generator = ReportGenerator.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )

    # Load test data
    config = generator.model.config
    test_csv = args.test_csv or str(
        Path(config.data.processed_dir) / "test_split.csv"
    )
    logger.info(f"Loading test data from {test_csv}")
    test_df = pd.read_csv(test_csv)

    if args.num_samples:
        test_df = test_df.head(args.num_samples)

    logger.info(f"Evaluating on {len(test_df)} samples...")

    # Generate reports
    references = []
    hypotheses = []
    samples = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating"):
        ref = row["report"]
        hyp = generator.predict(row["image_path"])

        references.append(ref)
        hypotheses.append(hyp)

        samples.append({
            "image_path": row["image_path"],
            "reference": ref,
            "generated": hyp,
        })

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(references, hypotheses)
    logger.info("\n" + format_metrics(metrics))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "metrics": metrics,
        "num_samples": len(test_df),
        "checkpoint": args.checkpoint,
        "samples": samples[:20],  # save first 20 examples
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
