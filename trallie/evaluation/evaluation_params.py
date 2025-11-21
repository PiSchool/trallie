"""
Dataset parameters and utilities for evaluation
"""
import json
from pathlib import Path
from typing import Dict, Any, List


DATASET_DESCRIPTIONS = {
    "fda_510ks": "FDA 510(k) medical device clearance documents",
    "wiki_nba_players": "NBA player profiles from Wikipedia",
    "enron": "Enron email corpus",
    "swde_movie_allmovie": "Movie information from AllMovie website",
    "swde_movie_amctv": "Movie information from AMC TV website",
    "swde_movie_hollywood": "Movie information from Hollywood.com website",
    "swde_movie_iheartmovies": "Movie information from iHeartMovies website",
    "swde_movie_imdb": "Movie information from IMDb website",
    "swde_movie_metacritic": "Movie information from Metacritic website",
    "swde_movie_rottentomatoes": "Movie information from Rotten Tomatoes website",
    "swde_movie_yahoo": "Movie information from Yahoo Movies website",
    "swde_university_collegeprowler": "University information from CollegeProwler website",
    "swde_university_ecampustours": "University information from eCampusTours website",
    "swde_university_embark": "University information from Embark website",
    "swde_university_matchcollege": "University information from MatchCollege website",
    "swde_university_usnews": "University information from US News website",
}


def get_evaluation_params(base_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Get evaluation parameters for all datasets.
    
    Args:
        base_path: Base path to the datasets
        
    Returns:
        Dictionary mapping dataset names to their parameters
    """
    datasets = {}
    
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        table_file = dataset_dir / "table.json"
        docs_dir = dataset_dir / "docs"
        
        if not table_file.exists():
            continue
        
        # Check if docs are extracted
        if docs_dir.exists():
            # Find all document files
            doc_files = list(docs_dir.glob("*"))
            if doc_files:
                files_pattern = str(docs_dir / "*")
            else:
                files_pattern = None
        else:
            files_pattern = None
        
        datasets[dataset_name] = {
            "ground_truth": str(table_file),
            "description": DATASET_DESCRIPTIONS.get(dataset_name, f"A dataset of {dataset_name}"),
            "files": files_pattern,
            "docs_dir": str(docs_dir) if docs_dir.exists() else None,
            "dataset_dir": str(dataset_dir)
        }
    
    return datasets


def get_dataset_schema(ground_truth_path: str) -> List[str]:
    """
    Extract schema (attribute names) from ground truth file.
    
    Args:
        ground_truth_path: Path to ground truth JSON file
        
    Returns:
        List of attribute names
    """
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    # Get attributes from first document
    if ground_truth:
        first_doc = next(iter(ground_truth.values()))
        return list(first_doc.keys())
    
    return []


def extract_docs_from_tar(dataset_dir: Path) -> bool:
    """
    Extract documents from tar.gz file if needed.
    
    Args:
        dataset_dir: Path to dataset directory
        
    Returns:
        True if extraction successful or docs already exist
    """
    import tarfile
    
    docs_dir = dataset_dir / "docs"
    tar_file = dataset_dir / "docs.tar.gz"
    
    # Check if already extracted
    if docs_dir.exists() and list(docs_dir.glob("*")):
        return True
    
    # Extract if tar file exists
    if tar_file.exists():
        print(f"Extracting {tar_file}...")
        try:
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(path=dataset_dir)
            print(f"Extracted to {docs_dir}")
            return True
        except Exception as e:
            print(f"Error extracting {tar_file}: {e}")
            return False
    
    return False

