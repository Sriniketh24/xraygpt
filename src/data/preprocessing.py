"""
Image and text preprocessing for the IU X-Ray dataset.

Handles:
  - Image loading and transforms (resize, normalize for ViT)
  - Report text cleaning and normalization
  - Train/val/test splitting
"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer


class XRayPreprocessor:
    """Preprocesses chest X-ray images and radiology reports."""

    # ImageNet normalization (used by ViT pretrained on ImageNet)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        image_size: int = 224,
        max_length: int = 128,
        tokenizer_name: str = "gpt2",
    ):
        self.image_size = image_size
        self.max_length = max_length

        # Image transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        # GPT-2 has no pad token by default — use eos as pad
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_image(self, image_path: str) -> Image.Image:
        """Load an image and convert to RGB."""
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def transform_image(self, image: Image.Image, is_train: bool = True) -> torch.Tensor:
        """Apply transforms to an image."""
        transform = self.train_transform if is_train else self.eval_transform
        return transform(image)

    @staticmethod
    def clean_report(text: str) -> str:
        """Clean and normalize a radiology report."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove non-printable characters
        text = re.sub(r"[^\x20-\x7E]", "", text)
        # Normalize common abbreviations
        text = text.replace("x-ray", "x-ray").replace("X-RAY", "x-ray")
        # Remove very short or empty reports
        if len(text.split()) < 3:
            return ""
        return text

    def tokenize_report(
        self, text: str, max_length: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        """Tokenize a report with padding and truncation."""
        max_len = max_length or self.max_length
        encoded = self.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def parse_iu_xray_reports(data_dir: str) -> pd.DataFrame:
    """
    Parse IU X-Ray dataset reports and build a DataFrame.

    Expected directory structure:
        data_dir/
            reports/     (JSON files from Open-i API)
            images/      (PNG images)

    Returns a DataFrame with columns:
        uid, image_path, findings, impression, report
    """
    data_path = Path(data_dir)
    reports_dir = data_path / "reports"
    images_dir = data_path / "images"

    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

    records = []

    for report_file in sorted(reports_dir.glob("*.json")):
        try:
            with open(report_file) as f:
                item = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        uid = item.get("uid", report_file.stem)

        # Extract report sections
        findings = ""
        impression = ""

        # The Open-i API returns abstract text which contains findings/impression
        abstract = item.get("abstract", "")
        if isinstance(abstract, str):
            # Try to split into findings and impression
            lower = abstract.lower()
            if "findings:" in lower and "impression:" in lower:
                parts = re.split(r"(?i)(findings:|impression:)", abstract)
                for i, part in enumerate(parts):
                    if part.lower().strip() == "findings:" and i + 1 < len(parts):
                        findings = parts[i + 1].strip()
                    elif part.lower().strip() == "impression:" and i + 1 < len(parts):
                        impression = parts[i + 1].strip()
            else:
                findings = abstract.strip()

        # Also check MeSH/abstract fields
        if not findings:
            findings = item.get("findings", "")
        if not impression:
            impression = item.get("impression", "")

        # Combine for full report
        report_parts = []
        if findings:
            report_parts.append(f"Findings: {findings}")
        if impression:
            report_parts.append(f"Impression: {impression}")

        report = " ".join(report_parts).strip()
        if not report:
            continue

        # Find associated images
        image_refs = []
        for img_field in ["imgLarge", "imgGrid150"]:
            img_val = item.get(img_field, "")
            if img_val:
                image_refs.extend([x.strip() for x in img_val.split(",") if x.strip()])

        # Match to local image files
        matched_images = []
        for img_ref in image_refs:
            img_name = img_ref.split("/")[-1] if "/" in img_ref else img_ref
            img_name_clean = re.sub(r"[^\w\-.]", "_", img_name)
            img_path = images_dir / img_name_clean
            if img_path.exists():
                matched_images.append(str(img_path))

        # If no matched images, try to find by UID pattern
        if not matched_images:
            for img_path in images_dir.glob(f"*{uid}*"):
                matched_images.append(str(img_path))

        # Create one record per image-report pair
        for img_path in matched_images:
            records.append({
                "uid": uid,
                "image_path": img_path,
                "findings": XRayPreprocessor.clean_report(findings),
                "impression": XRayPreprocessor.clean_report(impression),
                "report": XRayPreprocessor.clean_report(report),
            })

    df = pd.DataFrame(records)

    # Drop entries with empty reports
    df = df[df["report"].str.len() > 0].reset_index(drop=True)

    return df


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test by UID to prevent data leakage.

    Important: We split by patient UID so that images from the same patient
    don't appear in both train and test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Get unique UIDs
    uids = df["uid"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(uids)

    n = len(uids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_uids = set(uids[:train_end])
    val_uids = set(uids[train_end:val_end])
    test_uids = set(uids[val_end:])

    train_df = df[df["uid"].isin(train_uids)].reset_index(drop=True)
    val_df = df[df["uid"].isin(val_uids)].reset_index(drop=True)
    test_df = df[df["uid"].isin(test_uids)].reset_index(drop=True)

    return train_df, val_df, test_df
