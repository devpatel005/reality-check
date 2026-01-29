# Reality Check - Backend Data Pipeline

A headless Python application that scrapes prediction market data from Kalshi, generates semantic embeddings, and maintains a `markets.json` data file for use by the Reality Check Chrome extension.

## Overview

This pipeline:
1. Fetches all open events from the Kalshi prediction market API
2. Converts event titles into 384-dimensional vectors using `all-MiniLM-L6-v2`
3. Saves the data to `output/markets.json` with minimized JSON
4. Runs automatically via GitHub Actions every hour

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Initial Setup (Bootstrap)

Run the bootstrap script once to generate the initial `markets.json`:

```bash
python scripts/bootstrap/generate_initial.py
```

### Incremental Updates

For subsequent updates (used by GitHub Actions):

```bash
python main.py
```

## Data Schema

The `output/markets.json` file contains an array of market objects with minimized keys:

| Key | Type | Description |
|-----|------|-------------|
| `id` | string | The Kalshi event ticker |
| `t` | string | Cleaned event title |
| `p` | float | Current "Yes" probability (0-1) |
| `v` | array | 384-dimension float vector |
| `s` | string | Source identifier ("K" for Kalshi) |

## Project Structure

```
Reality-Check/
├── .github/
│   └── workflows/
│       └── update.yml          # Hourly cron job
├── scripts/
│   └── bootstrap/
│       └── generate_initial.py # One-time initial setup
├── output/
│   └── markets.json            # Generated data file
├── kalshi_provider.py          # Kalshi API client
├── main.py                     # Incremental update engine
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## GitHub Actions

The workflow runs automatically every hour and:
- Caches the ~80MB model files to speed up runs
- Uses delta logic to only compute new vectors
- Auto-commits changes to the repository
- Sends email notifications on failure (GitHub default)

To manually trigger an update, go to Actions → "Update Markets Data" → "Run workflow".

## License

MIT
