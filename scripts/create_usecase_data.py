import os
import subprocess


def install_requirements():
    """
    Install required packages for the script to run.
    """
    required_packages = ["datasets", "sec-edgar-downloader"]
    for package in required_packages:
        try:
            subprocess.check_call(["pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install package {package}. Error: {e}")


def save_to_text_files(dataset, output_dir="billsum_data", key="text"):
    """
    Save entries from a dataset to individual text files.

    Args:
        dataset: The dataset to save.
        output_dir: Directory to save the text files.
        key: Key in the dataset dictionary to save.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, example in enumerate(dataset):
        file_path = os.path.join(output_dir, f"text_{i}.txt")
        with open(file_path, "w") as f:
            f.write(example[key])


def create_raw_text_data(dataset_name, num_entries, output_dir, split, key):
    """
    Load a dataset and save it as text files.

    Args:
        dataset_name: Name of the dataset to load.
        num_entries: Number of entries to process.
        output_dir: Directory to save the text files.
        key: Key in the dataset dictionary to save.
    """
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.select(range(num_entries))
    save_to_text_files(dataset, output_dir, key)


def create_sec_filing_data(filing_type, company_ticker, output_dir="sec_filings"):
    """
    Download SEC filings for a specific company.

    Args:
        filing_type: Type of SEC filing (e.g., "8-K").
        company_ticker: Ticker symbol of the company (e.g., "AAPL").
        output_dir: Directory to save the downloaded filings.
    """
    from sec_edgar_downloader import Downloader

    print(f"Downloading {filing_type} filings for {company_ticker}")
    dl = Downloader(output_dir, "my.email@domain.com")
    dl.get(filing_type, company_ticker)


def main():
    install_requirements()

    # Dataset configurations
    datasets = [
        {
            "name": "FiscalNote/billsum",
            "entries": 10,
            "output_dir": "./data/raw/billsum_data",
            "split": "train",
            "key": "text",
        },
        {
            "name": "darrow-ai/LegalLensNLI",
            "entries": 10,
            "output_dir": "./data/raw/legallens_data",
            "split": "train",
            "key": "premise",
        },
        {
            "name": "NortheasternUniversity/big_patent",
            "entries": 10,
            "output_dir": "./data/raw/big_patent_data",
            "split": "train",
            "key": "description",
        },
    ]

    # Process each dataset
    for dataset in datasets:
        create_raw_text_data(
            dataset_name=dataset["name"],
            num_entries=dataset["entries"],
            output_dir=dataset["output_dir"],
            split=dataset["split"],
            key=dataset["key"],
        )

    # SEC filing configurations
    sec_filings = [
        {"filing_type": "8-K", "company_ticker": "AAPL"},
        {"filing_type": "10-K", "company_ticker": "MSFT"},
    ]

    # Download SEC filings
    for filing in sec_filings:
        create_sec_filing_data(
            filing_type=filing["filing_type"],
            company_ticker=filing["company_ticker"],
            output_dir="sec_filings",
        )


if __name__ == "__main__":
    main()
