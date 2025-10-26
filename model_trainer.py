import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

class MatchPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.team_encoder = LabelEncoder()
        self.is_trained = False
        
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Create copy to avoid modifying original data
        data = df.copy()
        
        # Encode team names
        all_teams = pd.concat([data['home_team'], data['away_team']]).unique()
        self.team_encoder.fit(all_teams)
        
        data['home_team_encoded'] = self.team_encoder.transform(data['home_team'])
        data['away_team_encoded'] = self.team_encoder.transform(data['away_team'])
        
        # Feature engineering
        data['total_shots'] = data['home_shots'] + data['away_shots']
        data['shot_ratio_home'] = data['home_shots'] / (data['total_shots'] + 1e-5)
        data['possession_ratio_home'] = data['home_possession'] / (data['home_possession'] + data['away_possession'] + 1e-5)
        data['pass_accuracy_diff'] = data['home_pass_accuracy'] - data['away_pass_accuracy']
        
        # Select features for model
        feature_columns = [
            'home_team_encoded', 'away_team_encoded',
            'home_shots', 'away_shots', 
            'home_shots_on_target', 'away_shots_on_target',
            'home_possession', 'away_possession',
            'home_pass_accuracy', 'away_pass_accuracy',
            'home_fouls', 'away_fouls',
            'home_corners', 'away_corners',
            'shot_ratio_home', 'possession_ratio_home', 'pass_accuracy_diff'
        ]
        
        X = data[feature_columns]
        y = data['outcome']
        
        return X, y, feature_columns
    
    def train(self, df, model_type='random_forest'):
        """Train the prediction model"""
        X, y, feature_columns = self.prepare_features(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                objective='multi:softprob'
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        self.feature_columns = feature_columns
        
        return accuracy, X_test_scaled, y_test, y_pred
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Predict outcome for a specific match"""
        if not self.is_trained:
            raise Exception("Model must be trained first")
        
        # Prepare features for prediction
        features = self.create_prediction_features(
            home_team, away_team, home_stats, away_stats
        )
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probabilities
    
    def create_prediction_features(self, home_team, away_team, home_stats, away_stats):
        """Create feature array for prediction"""
        # Encode teams
        home_encoded = self.team_encoder.transform([home_team])[0]
        away_encoded = self.team_encoder.transform([away_team])[0]
        
        # Calculate derived features
        total_shots = home_stats['shots'] + away_stats['shots']
        shot_ratio_home = home_stats['shots'] / (total_shots + 1e-5)
        possession_ratio_home = home_stats['possession'] / (home_stats['possession'] + away_stats['possession'] + 1e-5)
        pass_accuracy_diff = home_stats['pass_accuracy'] - away_stats['pass_accuracy']
        
        features = [
            home_encoded, away_encoded,
            home_stats['shots'], away_stats['shots'],
            home_stats['shots_on_target'], away_stats['shots_on_target'],
            home_stats['possession'], away_stats['possession'],
            home_stats['pass_accuracy'], away_stats['pass_accuracy'],
            home_stats['fouls'], away_stats['fouls'],
            home_stats['corners'], away_stats['corners'],
            shot_ratio_home, possession_ratio_home, pass_accuracy_diff
        ]
        
        return features
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'team_encoder': self.team_encoder,
                'feature_columns': self.feature_columns
            }, filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        loaded = joblib.load(filepath)
        self.model = loaded['model']
        self.scaler = loaded['scaler']
        self.team_encoder = loaded['team_encoder']
        self.feature_columns = loaded['feature_columns']
        self.is_trained = True
        print(f"Model loaded from {filepath}")