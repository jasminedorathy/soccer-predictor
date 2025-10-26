import os
from dotenv import load_dotenv

load_dotenv()

# Get your API key from https://rapidapi.com/api-sports/api/api-football/
API_KEY = os.getenv('API_FOOTBALL_KEY', 'your-api-key-here')
BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# League and season configuration
LEAGUE_ID = 39  # Premier League
SEASON = 2024
