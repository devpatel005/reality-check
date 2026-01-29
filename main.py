"""
Reality Check - Main Update Script

Performs incremental updates to markets.json using delta logic:
- New markets: Generate vectors
- Existing markets: Update probability only
- Expired markets: Remove from dataset

This script is designed to be run by GitHub Actions on a schedule.
"""

import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from kalshi_provider import KalshiProvider


# Constants
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "markets.json"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_PRECISION = 5  # Decimal places for vector floats


def round_vector(vector: list[float], precision: int = VECTOR_PRECISION) -> list[float]:
    """Round vector values to specified decimal places."""
    return [round(v, precision) for v in vector]


def load_existing_data() -> dict:
    """
    Load existing markets.json into a dictionary keyed by ID.
    
    Returns:
        Dictionary mapping market ID to market data.
    """
    if not OUTPUT_FILE.exists():
        return {}
    
    try:
        with open(OUTPUT_FILE, "r") as f:
            data = json.load(f)
        return {item["id"]: item for item in data}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_data(markets: list[dict]) -> None:
    """
    Save markets data to JSON with minimized output.
    
    Args:
        markets: List of market dictionaries.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(markets, f, separators=(",", ":"))


def main():
    # Step 1: Fetch current events from Kalshi
    provider = KalshiProvider()
    
    try:
        current_events = provider.fetch_all_events()
    except RuntimeError as e:
        print(f"ERROR: Failed to fetch events: {e}")
        sys.exit(1)
    
    # Safety check: Don't wipe data on empty API response
    if not current_events:
        print("WARNING: API returned 0 events. Exiting without changes.")
        sys.exit(0)
    
    # Step 2: Load existing data
    old_data = load_existing_data()
    
    # Step 3: Identify new, updated, and expired markets
    current_ids = {event["id"] for event in current_events}
    old_ids = set(old_data.keys())
    
    new_ids = current_ids - old_ids
    updated_ids = current_ids & old_ids
    expired_ids = old_ids - current_ids
    
    # Step 4: Load model only if we have new markets
    model = None
    if new_ids:
        model = SentenceTransformer(MODEL_NAME)
    
    # Step 5: Process events
    final_data = []
    new_events_for_embedding = []
    
    for event in current_events:
        event_id = event["id"]
        
        if event_id in new_ids:
            # New market - queue for embedding
            new_events_for_embedding.append(event)
        else:
            # Existing market - update probability, keep existing vector
            old_entry = old_data[event_id]
            final_data.append({
                "id": event_id,
                "t": event["title"],
                "p": round(event["prob"], 4),
                "v": old_entry["v"],  # Keep existing vector
                "s": "K"
            })
    
    # Step 6: Generate vectors for new markets
    if new_events_for_embedding:
        titles = [event["title"] for event in new_events_for_embedding]
        vectors = model.encode(titles)
        
        for event, vector in zip(new_events_for_embedding, vectors):
            final_data.append({
                "id": event["id"],
                "t": event["title"],
                "p": round(event["prob"], 4),
                "v": round_vector(vector.tolist()),
                "s": "K"
            })
    
    # Step 7: Save the final data
    save_data(final_data)
    
    # Output summary for GitHub Actions logs
    print(f"Update complete: {len(new_ids)} new, {len(updated_ids)} updated, {len(expired_ids)} expired")
    print(f"Total markets: {len(final_data)}")


if __name__ == "__main__":
    main()
