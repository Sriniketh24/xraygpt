"""
PyTorch Dataset for IU X-Ray image-report pairs.
"""

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import XRayPreprocessor


class IUXRayDataset(Dataset):
    """
    Dataset for paired chest X-ray images and radiology reports.

    Each sample returns:
        - pixel_values: preprocessed image tensor (3, H, W)
        - input_ids: tokenized report (max_length,)
        - attention_mask: token attention mask (max_length,)
        - report_text: raw report string (for evaluation)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        preprocessor: XRayPreprocessor,
        is_train: bool = True,
        max_length: Optional[int] = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.is_train = is_train
        self.max_length = max_length or preprocessor.max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]

        # Load and transform image
        image = self.preprocessor.load_image(row["image_path"])
        pixel_values = self.preprocessor.transform_image(image, is_train=self.is_train)

        # Tokenize report
        report_text = row["report"]
        tokens = self.preprocessor.tokenize_report(report_text, max_length=self.max_length)

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "report_text": report_text,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    """Custom collate function that handles mixed tensor/string data."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    report_texts = [item["report_text"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "report_texts": report_texts,
    }
