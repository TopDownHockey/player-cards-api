#!/bin/bash

# Replace YOUR_RAILWAY_URL with your actual Railway URL
RAILWAY_URL="${1:-https://your-app-name.up.railway.app}"

echo "Testing Railway API at: $RAILWAY_URL"
echo ""

# Test 1: Health check
echo "1. Testing root endpoint (/):"
curl -s "$RAILWAY_URL/" | python3 -m json.tool
echo ""

# Test 2: Health endpoint
echo "2. Testing /health endpoint:"
curl -s "$RAILWAY_URL/health" | python3 -m json.tool
echo ""

# Test 3: Live games endpoint
echo "3. Testing /api/live-games endpoint:"
curl -s "$RAILWAY_URL/api/live-games" | python3 -m json.tool | head -50
echo ""

# Test 4: Play-by-play endpoint (you'll need to replace with actual game_id)
echo "4. Testing /api/live-games-pbp endpoint (replace game_id):"
echo "Example: curl '$RAILWAY_URL/api/live-games-pbp?game_id=2024020001&home_team=TOR&away_team=MTL'"
curl -s "$RAILWAY_URL/api/live-games-pbp?game_id=2024020001&home_team=TOR&away_team=MTL" | python3 -m json.tool | head -30
echo ""

echo "Done!"

