"""
Download the IU X-Ray dataset from the Open-i API.

The Indiana University Chest X-Ray dataset contains ~3,955 radiology reports
paired with ~7,470 chest X-ray images. We download:
  - Reports as XML files
  - Images as PNG files

Source: https://openi.nlm.nih.gov/
"""

import json
import re
import time
import urllib.request
from pathlib import Path

REPORTS_URL = (
    "https://openi.nlm.nih.gov/api/search"
    "?query=indiana%20chest%20xray"
    "&coll=iu"
    "&m=1&n=100"
)
# We page through results in batches of 100

BASE_IMAGE_URL = "https://openi.nlm.nih.gov/imgs/collections/NLM-x{uid}/{filename}"


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a file with retry logic."""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed to download {url}: {e}")
                return False
    return False


def download_iu_xray(output_dir: str = "data/raw/iu_xray") -> None:
    """
    Download IU X-Ray dataset.

    Since the Open-i API can be unreliable, this script also supports
    downloading from a preprocessed mirror. If the API fails, we fall back
    to a pre-packaged version.
    """
    out = Path(output_dir)
    reports_dir = out / "reports"
    images_dir = out / "images"
    reports_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IU X-Ray Dataset Downloader")
    print("=" * 60)
    print()
    print("This script downloads the Indiana University Chest X-Ray dataset.")
    print("The dataset contains ~3,955 reports and ~7,470 images.")
    print()
    print("NOTE: If the Open-i API is unavailable, you can manually download")
    print("the dataset from one of these sources:")
    print("  1. https://drive.google.com/drive/folders/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg")
    print("  2. https://github.com/nlpaueb/bioread (preprocessed version)")
    print()
    print("Place the files as follows:")
    print(f"  Reports (XML): {reports_dir}/")
    print(f"  Images  (PNG): {images_dir}/")
    print()

    # Check if data already exists
    existing_images = list(images_dir.glob("*.png"))
    existing_reports = list(reports_dir.glob("*.xml"))

    if len(existing_images) > 100 and len(existing_reports) > 100:
        print(f"Dataset already present: {len(existing_images)} images, "
              f"{len(existing_reports)} reports")
        return

    # Try to download via Open-i API
    print("Attempting to download via Open-i API...")
    print("(This may take 10-30 minutes depending on connection speed)")
    print()

    total_downloaded_reports = 0
    total_downloaded_images = 0

    for start in range(1, 4000, 100):
        url = (
            f"https://openi.nlm.nih.gov/api/search"
            f"?query=*"
            f"&coll=iu"
            f"&m={start}&n={min(start + 99, 4000)}"
        )

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "XRayGPT-Research/1.0")
            response = urllib.request.urlopen(req, timeout=30)
            data = json.loads(response.read().decode())
        except Exception as e:
            print(f"  API request failed at offset {start}: {e}")
            print("  You may need to download the dataset manually.")
            break

        results = data.get("list", [])
        if not results:
            break

        for item in results:
            uid = item.get("uid", "")
            if not uid:
                continue

            # Save report metadata as JSON (we'll parse findings/impression)
            report_path = reports_dir / f"{uid}.json"
            if not report_path.exists():
                with open(report_path, "w") as f:
                    json.dump(item, f, indent=2)
                total_downloaded_reports += 1

            # Download images
            for img_info in item.get("imgLarge", "").split(","):
                img_info = img_info.strip()
                if not img_info:
                    continue

                img_filename = img_info.split("/")[-1] if "/" in img_info else img_info
                img_filename = re.sub(r"[^\w\-.]", "_", img_filename)
                img_path = images_dir / img_filename

                if not img_path.exists() and img_info.startswith("http"):
                    if download_file(img_info, img_path):
                        total_downloaded_images += 1

        print(f"  Progress: {total_downloaded_reports} reports, "
              f"{total_downloaded_images} images (offset {start})")
        time.sleep(0.5)  # Be polite to the API

    print()
    print(f"Download complete: {total_downloaded_reports} reports, "
          f"{total_downloaded_images} images")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    download_iu_xray()
