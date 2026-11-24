
from pathlib import Path
import urllib.request
import ssl
import certifi
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
CSV_PATH = DATASET_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"

ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/967/"
    "phiusiil+phishing+url+dataset.zip"
)


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists():
        print("Dataset already exists!")
        print("Rows:", len(pd.read_csv(CSV_PATH)))
        return

    zip_path = DATASET_DIR / "phiusiil_dataset.zip"

    print("Preparing secure UCI connection...")

    context = ssl.create_default_context(
        cafile=certifi.where()
    )

    print("Downloading official UCI dataset...")

    try:
        request = urllib.request.Request(
            ZIP_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(
            request,
            context=context,
            timeout=120
        ) as response:
            with open(zip_path, "wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)

        print("Download completed!")
        print("Extracting dataset...")

        with zipfile.ZipFile(zip_path, "r") as archive:
            csv_files = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                raise RuntimeError("No CSV file found in archive")

            with archive.open(csv_files[0]) as source:
                with open(CSV_PATH, "wb") as destination:
                    destination.write(source.read())

        zip_path.unlink(missing_ok=True)

        df = pd.read_csv(CSV_PATH)

        print("\nDataset downloaded successfully!")
        print("Total records:", len(df))
        print("Total columns:", len(df.columns))
        print("Saved:", CSV_PATH)

    except Exception as error:
        print("\nDownload failed!")
        print("Reason:", error)
        print("You can retry after checking your internet connection.")


if __name__ == "__main__":
    main()
