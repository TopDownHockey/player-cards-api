from flask import Flask
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Import route handlers
from routes.live_games import live_games_route
from routes.live_games_pbp import live_games_pbp_route
from routes.test import test_route

@app.route("/")
def home():
    return {"status": "healthy", "message": "Hockey Stats API", "version": "1.0.0"}

@app.route("/health")
def health():
    return {"status": "ok"}

# Register routes
app.add_url_rule("/api/live-games", "live_games", live_games_route, methods=["GET"])
app.add_url_rule("/api/live-games-pbp", "live_games_pbp", live_games_pbp_route, methods=["GET"])
app.add_url_rule("/api/test", "test", test_route, methods=["GET"])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

