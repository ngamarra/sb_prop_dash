# Prop Bet Live Scorer (Streamlit)

Core Streamlit app for scoring participant prop sheets against a live Google Sheets answer key.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data Inputs

### 1) Master CSV

Expected columns (flexible names are supported):
- `question_id` (required)
- `description` (optional but recommended)
- `choices` (optional but recommended)

Default 2026 path in the app is `data/2026/props_master.csv`.

### 2) Answer Key (Google Sheet URL)

Put the published Google Sheet URL in the sidebar.  
The app tries these formats automatically:
- original URL
- Google export CSV URL
- Google gviz CSV URL
- HTML table parsing fallback

You can also use a local answer file (`.csv` or `.xlsx`) from the sidebar.
This is useful if your Google Sheet is synced/exported as `Answers 2026.xlsx`.

### 3) Participant CSV files

Each participant file should include at least:
- question column (`question_id`, `qid`, etc.)
- pick column (`pick`, `selection`, `choice`, etc.)

Optional:
- participant name column (`participant`, `name`, etc.)

If no participant name column is present, the filename is used.

## Running Modes

In the sidebar choose participant source:
- `Folder`: load all `*.csv` from a folder (default `participants/`)
- `Uploads`: upload one or more participant CSV files directly

## Live Refresh

Use sidebar controls:
- `Auto-refresh`: enable/disable polling
- `Auto-refresh interval (seconds)`: refresh frequency (5-300s)
- `Refresh now`: manual immediate refresh

When answer source is `Local answer file`, each refresh cycle re-reads that path/upload.

## Multi-Year Tabs

The app is tabbed by year. Configure defaults in:
- `/Users/ngamarra/Documents/GitHub/sb_prop_dash/app.py` (`YEAR_CONFIGS`)

Each year tab can point at its own:
- master questions CSV
- Google Sheet URL or local answer file
- participant folder (recommended: `participants/<year>`)

## Output

- Live leaderboard (`correct`, `incorrect`, `pending`, `no_pick`, `accuracy`, `completion`)
- Participant-level detail table
- CSV downloads for:
  - participant template
  - leaderboard
- Visualizations (points, status breakdown, accuracy vs completion)
- Analytics:
  - Monte Carlo win probability
  - leverage props (highest potential rank impact)
  - scenario simulator (force one unresolved prop)
