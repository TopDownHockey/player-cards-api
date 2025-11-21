# Player Cards API

Flask API for hockey statistics, migrated from Vercel serverless functions to Railway.

## Endpoints

- `GET /` - Health check
- `GET /health` - Health check endpoint
- `GET /api/live-games` - Get live game data with win probability simulations
- `GET /api/live-games-pbp` - Get play-by-play data with xG models

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run locally:
```bash
python main.py
```

Or with gunicorn:
```bash
gunicorn main:app --bind 0.0.0.0:5000 --timeout 300
```

## Deployment to Railway

1. Create a new GitHub repository for this code
2. Push the code to GitHub
3. In Railway, create a new project and connect to the GitHub repository
4. Railway will automatically detect Python and use the `requirements.txt` and `Procfile`
5. Set any required environment variables in Railway dashboard

## Environment Variables

- `PORT` - Set automatically by Railway
- `GITHUB_TOKEN` - If needed for TopDownHockey packages (set in Railway dashboard)

## Project Structure

```
player-cards-api/
├── main.py              # Main Flask app entry point
├── requirements.txt     # Python dependencies
├── Procfile             # Railway start command
├── railway.json         # Railway configuration
├── routes/
│   ├── live_games.py    # Live games route
│   └── live_games_pbp.py # Play-by-play route
├── models/              # ONNX ML models
└── data/                # CSV data files
```

## Notes

- Models are loaded once at startup (faster subsequent requests)
- Data files are read from the `data/` directory
- The API uses CORS to allow cross-origin requests

