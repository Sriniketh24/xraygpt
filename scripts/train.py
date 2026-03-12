"""
Training script for XRayGPT.

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml --resume checkpoints/checkpoint_epoch_5.pt
"""

import argparse
from pathlib import Path

from src.data.dataset import IUXRayDataset
from src.data.preprocessing import XRayPreprocessor, create_splits, parse_iu_xray_reports
from src.models.xraygpt import XRayGPT
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train XRayGPT")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(config.training.seed)

    # Setup logging
    log_dir = Path(config.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("xraygpt", log_file=str(log_dir / "train.log"))

    logger.info("=" * 60)
    logger.info("XRayGPT Training")
    logger.info("=" * 60)

    # Parse dataset
    logger.info(f"Loading dataset from {config.data.data_dir}...")
    df = parse_iu_xray_reports(config.data.data_dir)
    logger.info(f"Total samples: {len(df)}")

    # Split data
    train_df, val_df, test_df = create_splits(
        df,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split,
        seed=config.training.seed,
    )
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Save test split for later evaluation
    processed_dir = Path(config.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(processed_dir / "test_split.csv", index=False)
    logger.info(f"Saved test split to {processed_dir / 'test_split.csv'}")

    # Create preprocessor and datasets
    preprocessor = XRayPreprocessor(
        image_size=config.data.image_size,
        max_length=config.data.max_length,
        tokenizer_name=config.text.model_name,
    )

    train_dataset = IUXRayDataset(train_df, preprocessor, is_train=True)
    val_dataset = IUXRayDataset(val_df, preprocessor, is_train=False)

    # Build model
    logger.info("Building model...")
    model = XRayGPT(config)

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        log_file=str(log_dir / "train.log"),
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    history = trainer.train()

    # Log final results
    logger.info("Training history:")
    for epoch, (tl, vl) in enumerate(
        zip(history["train_loss"], history["val_loss"]), 1
    ):
        logger.info(f"  Epoch {epoch}: train_loss={tl:.4f}, val_loss={vl:.4f}")


if __name__ == "__main__":
    main()
