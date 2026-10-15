from pathlib import Path
import json
from rag.pipeline.normalizer import normalize_verdict

def adapt_scraped_facts(scraped_list: list, source_name: str) -> dict:
    """
    Convert a list of scraped fact-check dicts into the
    Phase-3 JSON shape expected by UniversalNormalizer.

    Input format (each item):
    {
        "claim": "Modi announced ₹15000 for all",
        "verdict": "FAKE",
        "explanation": "No such scheme exists",
        "source_url": "https://altnews.in/...",
        "source_name": "Alt News",
        "category": "politics",
        "date": "2024-03-15"
    }

    Output: JSON shape with "domain", "results", etc.
    """
    entries = []
    for item in scraped_list:
        entries.append({
            "url": item.get("source_url", item.get("url", "")),
            "status": "extracted",
            "data": [{
                "claim": (
                    item.get("claim_review", {}).get("claim")
                    or item.get("title", "")
                ),
                "description": (
                    item.get("summary")
                    or item.get("body")
                    or item.get("explanation", "")
                    or item.get("description", "")
                ),
                "verdict": normalize_verdict(
                    item.get("verdict_normalized")
                    or item.get("verdict", "")
                ),
                "source_url": item.get("source_url", item.get("url", "")),
                "source_name": item.get("source", source_name),
                "category": item.get("category", "general"),
                "date": item.get("date", "")
            }]
        })

    return {
        "domain": f"fact-check-{source_name}",
        "total_processed": len(entries),
        "total_healed": 0,
        "failed": 0,
        "results": {"fact_check": entries}
    }

def adapt_json_file(json_path: str) -> dict:
    """
    Load a JSON file and convert to pipeline shape.
    Handles two formats:
    1. Already in correct shape (has "results" key) → return as-is
    2. Flat list of fact-check dicts → wrap using adapt_scraped_facts
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Already in correct shape
    if isinstance(data, dict) and "results" in data:
        return data

    # Flat list of fact-checks
    if isinstance(data, list):
        # Try to detect source name from filename
        source_name = Path(json_path).stem
        return adapt_scraped_facts(data, source_name=source_name)

    raise ValueError(f"Unrecognized JSON structure in {json_path}")
