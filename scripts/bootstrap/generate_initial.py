"""
Bootstrap Script for Reality Check

One-time script to generate the initial markets.json file
by fetching all events and generating vectors for all titles.

Usage:
    python scripts/bootstrap/generate_initial.py
"""

import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from kalshi_provider import KalshiProvider


# Constants
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "markets.json"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_PRECISION = 5  # Decimal places for vector floats


def round_vector(vector: list[float], precision: int = VECTOR_PRECISION) -> list[float]:
    """Round vector values to specified decimal places."""
    return [round(v, precision) for v in vector]


def main():
    print("=" * 50)
    print("Reality Check - Bootstrap Script")
    print("=" * 50)
    
    # Step 1: Load the embedding model
    print("\n[1/4] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"      Model '{MODEL_NAME}' loaded successfully.")
    
    # Step 2: Fetch all events from Kalshi
    print("\n[2/4] Fetching events from Kalshi API...")
    provider = KalshiProvider()
    events = provider.fetch_all_events()
    
    if not events:
        print("      ERROR: No events fetched from API. Exiting.")
        sys.exit(1)
    
    print(f"      Fetched {len(events)} open events.")
    
    # Step 3: Generate vectors for all titles
    print("\n[3/4] Generating embeddings for all titles...")
    titles = [event["title"] for event in events]
    vectors = model.encode(titles, show_progress_bar=True)
    print(f"      Generated {len(vectors)} vectors (dimension: {len(vectors[0])}).")
    
    # Step 4: Build the output data structure
    print("\n[4/4] Building and saving markets.json...")
    markets_data = []
    
    for event, vector in zip(events, vectors):
        markets_data.append({
            "id": event["id"],
            "t": event["title"],
            "p": round(event["prob"], 4),
            "v": round_vector(vector.tolist()),
            "s": "K"  # Source: Kalshi
        })
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save with minimized JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(markets_data, f, separators=(",", ":"))
    
    # Calculate file size
    file_size = OUTPUT_FILE.stat().st_size
    file_size_kb = file_size / 1024
    file_size_mb = file_size_kb / 1024
    
    print(f"      Saved to: {OUTPUT_FILE}")
    print(f"      File size: {file_size_kb:.1f} KB ({file_size_mb:.2f} MB)")
    print(f"      Total markets: {len(markets_data)}")
    
    print("\n" + "=" * 50)
    print("Bootstrap complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
