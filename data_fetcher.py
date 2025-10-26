import requests
import pandas as pd
import time
from config import HEADERS, BASE_URL, LEAGUE_ID, SEASON

class FootballDataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def make_request(self, endpoint, params):
        """Make API request with error handling"""
        try:
            response = self.session.get(f"{BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return None
    
    def get_league_standings(self):
        """Get current league standings"""
        params = {"league": LEAGUE_ID, "season": SEASON}
        data = self.make_request("standings", params)
        return data
    
    def get_teams(self):
        """Get all teams in the league"""
        params = {"league": LEAGUE_ID, "season": SEASON}
        data = self.make_request("teams", params)
        return data
    
    def get_fixtures(self, from_date=None, to_date=None):
        """Get fixtures for the league"""
        params = {
            "league": LEAGUE_ID, 
            "season": SEASON,
            "from": from_date,
            "to": to_date
        }
        data = self.make_request("fixtures", params)
        return data
    
    def get_fixture_statistics(self, fixture_id):
        """Get detailed statistics for a specific fixture"""
        params = {"fixture": fixture_id}
        data = self.make_request("fixtures/statistics", params)
        return data
    
    def get_fixture_events(self, fixture_id):
        """Get events (goals, cards, etc.) for a fixture"""
        params = {"fixture": fixture_id}
        data = self.make_request("fixtures/events", params)
        return data
    
    def get_team_statistics(self, team_id):
        """Get team statistics for the season"""
        params = {"team": team_id, "league": LEAGUE_ID, "season": SEASON}
        data = self.make_request("teams/statistics", params)
        return data
    
    def get_injuries(self, team_id=None):
        """Get player injuries"""
        params = {"league": LEAGUE_ID, "season": SEASON}
        if team_id:
            params["team"] = team_id
        data = self.make_request("injuries", params)
        return data
    
    def create_training_dataset(self, num_matches=200):
        """Create a comprehensive dataset for model training"""
        print("Fetching fixtures...")
        fixtures = self.get_fixtures()
        
        if not fixtures:
            print("No fixtures found. Using mock data for demo...")
            return self.generate_mock_data(num_matches)
        
        match_data = []
        processed = 0
        
        for fixture in fixtures[:num_matches]:
            fixture_id = fixture['fixture']['id']
            home_team = fixture['teams']['home']['name']
            away_team = fixture['teams']['away']['name']
            
            print(f"Processing {home_team} vs {away_team}...")
            
            # Get fixture statistics
            stats = self.get_fixture_statistics(fixture_id)
            events = self.get_fixture_events(fixture_id)
            
            if stats and events:
                match_features = self.extract_match_features(fixture, stats, events)
                if match_features:
                    match_data.append(match_features)
                    processed += 1
            
            # Rate limiting
            time.sleep(1)
            
            if processed >= num_matches:
                break
        
        if not match_data:
            print("No valid data from API. Using mock data for demo...")
            return self.generate_mock_data(num_matches)
        
        return pd.DataFrame(match_data)
    
    def extract_match_features(self, fixture, stats, events):
        """Extract features from match data"""
        try:
            # Basic match info
            home_team = fixture['teams']['home']['name']
            away_team = fixture['teams']['away']['name']
            home_goals = fixture['goals']['home'] or 0
            away_goals = fixture['goals']['away'] or 0
            
            # Determine outcome
            if home_goals > away_goals:
                outcome = 'home_win'
            elif away_goals > home_goals:
                outcome = 'away_win'
            else:
                outcome = 'draw'
            
            # Extract statistics
            home_stats = {}
            away_stats = {}
            
            for team_stats in stats:
                if team_stats['team']['name'] == home_team:
                    home_stats = self.parse_statistics(team_stats['statistics'])
                elif team_stats['team']['name'] == away_team:
                    away_stats = self.parse_statistics(team_stats['statistics'])
            
            # Extract events
            shots_home = len([e for e in events if e.get('team', {}).get('name') == home_team and e.get('type') == 'Shot'])
            shots_away = len([e for e in events if e.get('team', {}).get('name') == away_team and e.get('type') == 'Shot'])
            
            # Create feature dictionary
            features = {
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'outcome': outcome,
                'home_shots': home_stats.get('Total Shots', 0),
                'away_shots': away_stats.get('Total Shots', 0),
                'home_shots_on_target': home_stats.get('Shots on Goal', 0),
                'away_shots_on_target': away_stats.get('Shots on Goal', 0),
                'home_possession': home_stats.get('Ball Possession', 0),
                'away_possession': away_stats.get('Ball Possession', 0),
                'home_passes': home_stats.get('Total passes', 0),
                'away_passes': away_stats.get('Total passes', 0),
                'home_pass_accuracy': home_stats.get('Passes %', 0),
                'away_pass_accuracy': away_stats.get('Passes %', 0),
                'home_fouls': home_stats.get('Fouls', 0),
                'away_fouls': away_stats.get('Fouls', 0),
                'home_corners': home_stats.get('Corner Kicks', 0),
                'away_corners': away_stats.get('Corner Kicks', 0),
                'home_yellow_cards': home_stats.get('Yellow Cards', 0),
                'away_yellow_cards': away_stats.get('Yellow Cards', 0),
                'home_offsides': home_stats.get('Offsides', 0),
                'away_offsides': away_stats.get('Offsides', 0),
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def parse_statistics(self, statistics):
        """Parse statistics into a dictionary"""
        stats_dict = {}
        for stat in statistics:
            stats_dict[stat['type']] = self.safe_float(stat['value'])
        return stats_dict
    
    def safe_float(self, value):
        """Safely convert to float, handling percentage strings"""
        if value is None:
            return 0.0
        try:
            if isinstance(value, str) and '%' in value:
                return float(value.replace('%', ''))
            return float(value)
        except ValueError:
            return 0.0

    def generate_mock_data(self, num_matches=50):
        """Generate mock match data for demo purposes when API fails"""
        import random
        
        # Sample teams
        teams = ['Manchester United', 'Liverpool', 'Chelsea', 'Arsenal', 'Manchester City', 'Tottenham', 'Everton', 'Newcastle']
        
        mock_data = []
        for _ in range(num_matches):
            home_team = random.choice(teams)
            away_team = random.choice([t for t in teams if t != home_team])
            
            # Random goals and outcome
            home_goals = random.randint(0, 5)
            away_goals = random.randint(0, 5)
            if home_goals > away_goals:
                outcome = 'home_win'
            elif away_goals > home_goals:
                outcome = 'away_win'
            else:
                outcome = 'draw'
            
            # Random stats
            home_shots = random.randint(5, 20)
            away_shots = random.randint(5, 20)
            home_shots_on_target = random.randint(2, min(home_shots, 10))
            away_shots_on_target = random.randint(2, min(away_shots, 10))
            home_possession = random.randint(40, 60)
            away_possession = 100 - home_possession
            home_pass_accuracy = random.randint(70, 95)
            away_pass_accuracy = random.randint(70, 95)
            home_fouls = random.randint(5, 15)
            away_fouls = random.randint(5, 15)
            home_corners = random.randint(3, 10)
            away_corners = random.randint(3, 10)
            home_yellow_cards = random.randint(0, 4)
            away_yellow_cards = random.randint(0, 4)
            home_offsides = random.randint(0, 3)
            away_offsides = random.randint(0, 3)
            
            features = {
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'outcome': outcome,
                'home_shots': home_shots,
                'away_shots': away_shots,
                'home_shots_on_target': home_shots_on_target,
                'away_shots_on_target': away_shots_on_target,
                'home_possession': home_possession,
                'away_possession': away_possession,
                'home_pass_accuracy': home_pass_accuracy,
                'away_pass_accuracy': away_pass_accuracy,
                'home_fouls': home_fouls,
                'away_fouls': away_fouls,
                'home_corners': home_corners,
                'away_corners': away_corners,
                'home_yellow_cards': home_yellow_cards,
                'away_yellow_cards': away_yellow_cards,
                'home_offsides': home_offsides,
                'away_offsides': away_offsides,
            }
            mock_data.append(features)
        
        print(f"Generated {len(mock_data)} mock matches for demo.")
        return pd.DataFrame(mock_data)
