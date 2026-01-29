"""
Kalshi API Provider Module

Handles fetching open events from the Kalshi prediction market API
with cursor-based pagination and retry logic.
"""

import time
import requests
from typing import Optional


class KalshiProvider:
    """Client for fetching prediction market data from Kalshi API."""
    
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    EVENTS_ENDPOINT = "/events"
    
    def __init__(self, max_retries: int = 3, base_delay: float = 2.0):
        """
        Initialize the Kalshi provider.
        
        Args:
            max_retries: Maximum number of retry attempts for failed requests.
            base_delay: Base delay in seconds for exponential backoff.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "RealityCheck/1.0"
        })
    
    def _make_request(self, url: str, params: dict) -> Optional[dict]:
        """
        Make a request with retry logic and exponential backoff.
        
        Args:
            url: The URL to request.
            params: Query parameters.
            
        Returns:
            JSON response as dict, or None if all retries failed.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {e}")
        return None
    
    def _extract_probability(self, event: dict) -> float:
        """
        Extract the "Yes" probability from an event's nested markets.
        
        Args:
            event: The event dictionary from the API.
            
        Returns:
            Probability as a float between 0 and 1.
        """
        markets = event.get("markets", [])
        if not markets:
            return 0.0
        
        # Use the first market's data
        market = markets[0]
        
        # Try yes_price first (most accurate)
        if "yes_price" in market and market["yes_price"] is not None:
            # Kalshi prices are in cents (0-100), convert to probability
            return market["yes_price"] / 100.0
        
        # Fallback to mid-price from bid/ask
        yes_bid = market.get("yes_bid", 0) or 0
        yes_ask = market.get("yes_ask", 100) or 100
        return ((yes_bid + yes_ask) / 2) / 100.0
    
    def _clean_title(self, title: str) -> str:
        """
        Clean and normalize the event title.
        
        Args:
            title: Raw title from the API.
            
        Returns:
            Cleaned title string.
        """
        if not title:
            return ""
        # Remove extra whitespace
        return " ".join(title.split())
    
    def fetch_all_events(self) -> list[dict]:
        """
        Fetch all open events with cursor-based pagination.
        
        Returns:
            List of dictionaries with keys: 'id', 'title', 'prob'
        """
        url = f"{self.BASE_URL}{self.EVENTS_ENDPOINT}"
        params = {
            "status": "open",
            "limit": 200,
            "with_nested_markets": "true"
        }
        
        all_events = []
        cursor = None
        
        while True:
            if cursor:
                params["cursor"] = cursor
            elif "cursor" in params:
                del params["cursor"]
            
            data = self._make_request(url, params)
            
            if not data:
                break
            
            events = data.get("events", [])
            
            for event in events:
                event_ticker = event.get("event_ticker", "")
                title = event.get("title", "")
                
                if not event_ticker or not title:
                    continue
                
                all_events.append({
                    "id": event_ticker,
                    "title": self._clean_title(title),
                    "prob": self._extract_probability(event)
                })
            
            # Check for next page
            cursor = data.get("cursor")
            if not cursor:
                break
        
        return all_events


if __name__ == "__main__":
    # Quick test
    provider = KalshiProvider()
    events = provider.fetch_all_events()
    print(f"Fetched {len(events)} events")
    if events:
        print(f"Sample: {events[0]}")
