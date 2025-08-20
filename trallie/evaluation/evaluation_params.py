import json
from pathlib import Path

def get_evaluation_params(base_path):
    datasets = {
        "fda_510ks": {
            "ground_truth": base_path / "fda_510ks/table.json",
            "files": base_path / "fda_510ks/data/evaporate/fda-ai-pmas/510k/*.txt",
            "description": "FDA 510(k) submissions: A dataset of medical device approval documents submitted to the U.S. Food and Drug Administration.",
        },
        **{
            f"swde_movie_{site}": {
                "ground_truth": base_path / f"swde_movie_{site}/table.json",
                "files": base_path / f"swde_movie_{site}/data/evaporate/swde/movie/movie-{site}(2000)/*.htm",
                "description": f"SWDE Movie ({site.capitalize()}): Extracted movie dataset from {site.capitalize()} featuring metadata and summaries.",
            }
            for site in ["allmovie", "amctv", "hollywood", "iheartmovies", "imdb", "metacritic", "rottentomatoes", "yahoo"]
        },
        **{
            f"swde_university_{site}": {
                "ground_truth": base_path / f"swde_university_{site}/table.json",
                "files": base_path / f"swde_university_{site}/data/evaporate/swde/university/university-{site}({num})/*.htm",
                "description": f"SWDE University ({site.capitalize()}): Extracted university dataset from {site.capitalize()} covering various aspects.",
            }
            for site, num in [
                ("collegeprowler", 2000), ("ecampustours", 1063), ("embark", 2000),
                ("matchcollege", 2000), ("usnews", 1027)
            ]
        },
        "wiki_nba_players": {
            "ground_truth": base_path / "wiki_nba_players/table.json",
            "files": base_path / "wiki_nba_players/home/simran/fm-data-analysis/scratch/simran/wiki_nba_player_files.json",
            "description": "Wiki NBA Players: Dataset containing structured information about NBA players extracted from Wikipedia.",
        }
    }
    
    return datasets

def get_dataset_schema(table_path):
    """
    Extracts the unique schema (field names) from a dataset's table.json file.
    """
    if not table_path.exists():
        return f"Error: {table_path} not found"

    try:
        with open(table_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return "Error: Invalid JSON structure"

        schema = set()
        for record in data.values():
            if isinstance(record, dict):
                schema.update(record.keys())

        return sorted(schema)
    except json.JSONDecodeError:
        return "Error: Could not decode JSON"
