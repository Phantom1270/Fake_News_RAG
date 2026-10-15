"""
Ingest all scraped fact-check data into the RAG pipeline.

Prerequisites:
    Run scrapers first to populate data/scraped/ with JSON files.

Usage:
    python scripts/ingest_factchecks.py
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rag.pipeline.adapters import adapt_scraped_facts, adapt_json_file
from rag.pipeline.data_pipeline import DataPipeline
from rag.chunking.engine import ChunkingEngine
from rag.bootstrap import create_pipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest scraped fact-check data.")
    parser.add_argument("--input", type=str, default="", help="Path to a specific JSON file to ingest")
    parser.add_argument("--dir", type=str, default=os.path.join("scripts", "altnews_scraper", "data", "scraped"), help="Directory to scan for JSON files")
    args = parser.parse_args()

    output_dir = os.getenv("RAG_SCRAPER_OUTPUT__DIRECTORY", "./scraped_output")
    os.makedirs(output_dir, exist_ok=True)

    adapted_files = []
    files_to_process = []

    if args.input:
        if os.path.exists(args.input):
            files_to_process.append(args.input)
        else:
            print(f"Input file not found: {args.input}")
            return
    else:
        scraped_dir = args.dir
        if not os.path.exists(scraped_dir):
            print(f"No scraped data directory found at {scraped_dir}")
            print("Run scrapers first to collect fact-check data.")
            return
        for filename in sorted(os.listdir(scraped_dir)):
            if filename.endswith(".json"):
                files_to_process.append(os.path.join(scraped_dir, filename))

    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        print(f"Adapting {filename}...")

        try:
            adapted = adapt_json_file(filepath)
            out_path = os.path.join(output_dir, f"adapted_{filename}")
            with open(out_path, "w") as f:
                json.dump(adapted, f)
            adapted_files.append(out_path)
            print(f"  - {adapted['total_processed']} records")
        except Exception as e:
            print(f"  - Error: {e}")

    if not adapted_files:
        print("No valid data files found to ingest.")
        return

    # Run pipeline on all adapted files
    print(f"\nIngesting {len(adapted_files)} files into RAG pipeline...")
    pipeline = DataPipeline()

    total_docs = 0
    for filepath in adapted_files:
        print(f"\nProcessing {filepath}...")
        report = pipeline.run(filepath)
        doc_count = report.get("total_normalized", 0)
        total_docs += doc_count
        print(f"  Normalized: {report.get('total_normalized', 0)}")
        print(f"  Cleaned: {report.get('total_after_cleaning', 0)}")
        print(f"  Deduplicated: {report.get('total_after_dedup', 0)}")
        print(f"  Stored: {report.get('documents_stored', 0)}")

    print("\n=== Running Chunking Engine ===")
    chunker = ChunkingEngine()
    chunker.chunk_all_documents(save=True)

    print("\n=== Building Vector Index ===")
    rag = create_pipeline()
    indexer = rag["indexer"]
    indexer.build_all(force_rebuild=True)

    print(f"\n=== Full Ingestion Complete ===")
    print(f"Total raw documents normalized: {total_docs}")
    print(f"Database: {os.getenv('RAG_GENERAL__DATA_DIR', './data')}/rag.db")


if __name__ == "__main__":
    main()
