import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_fetcher import FootballDataFetcher
from model_trainer import MatchPredictor
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Soccer Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .team-stats {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SoccerPredictorApp:
    def __init__(self):
        self.fetcher = FootballDataFetcher()
        self.predictor = MatchPredictor()
        self.load_data()
    
    def load_data(self):
        """Load or initialize data"""
        if 'matches_df' not in st.session_state:
            st.session_state.matches_df = None
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
    
    def sidebar_controls(self):
        """Sidebar controls"""
        st.sidebar.title("⚽ Control Panel")
        
        # Data collection section
        st.sidebar.subheader("Data Collection")
        if st.sidebar.button("Fetch Match Data"):
            with st.spinner("Fetching match data from API..."):
                matches_df = self.fetcher.create_training_dataset(num_matches=50)
                if not matches_df.empty:
                    st.session_state.matches_df = matches_df
                    st.sidebar.success(f"Fetched {len(matches_df)} matches!")
                else:
                    st.sidebar.error("Failed to fetch data. Check API key.")
        
        # Model training section
        st.sidebar.subheader("Model Training")
        model_type = st.sidebar.selectbox(
            "Select Model",
            ["random_forest", "xgboost"]
        )
        
        if st.sidebar.button("Train Model") and st.session_state.matches_df is not None:
            with st.spinner("Training model..."):
                accuracy, X_test, y_test, y_pred = self.predictor.train(
                    st.session_state.matches_df, model_type
                )
                st.session_state.model_trained = True
                st.session_state.accuracy = accuracy
                st.sidebar.success(f"Model trained! Accuracy: {accuracy:.3f}")
        
        # Prediction section
        st.sidebar.subheader("Match Prediction")
        
        if st.session_state.matches_df is not None:
            teams = sorted(pd.concat([
                st.session_state.matches_df['home_team'], 
                st.session_state.matches_df['away_team']
            ]).unique())
            
            home_team = st.sidebar.selectbox("Home Team", teams)
            away_team = st.sidebar.selectbox("Away Team", [t for t in teams if t != home_team])
            
            st.sidebar.subheader("Home Team Stats")
            home_shots = st.sidebar.slider("Home Shots", 0, 30, 15)
            home_shots_target = st.sidebar.slider("Home Shots on Target", 0, 20, 8)
            home_possession = st.sidebar.slider("Home Possession (%)", 0, 100, 55)
            home_pass_accuracy = st.sidebar.slider("Home Pass Accuracy (%)", 0, 100, 80)
            home_fouls = st.sidebar.slider("Home Fouls", 0, 30, 12)
            home_corners = st.sidebar.slider("Home Corners", 0, 15, 6)
            
            st.sidebar.subheader("Away Team Stats")
            away_shots = st.sidebar.slider("Away Shots", 0, 30, 12)
            away_shots_target = st.sidebar.slider("Away Shots on Target", 0, 20, 6)
            away_possession = st.sidebar.slider("Away Possession (%)", 0, 100, 45)
            away_pass_accuracy = st.sidebar.slider("Away Pass Accuracy (%)", 0, 100, 75)
            away_fouls = st.sidebar.slider("Away Fouls", 0, 30, 14)
            away_corners = st.sidebar.slider("Away Corners", 0, 15, 4)
            
            home_stats = {
                'shots': home_shots,
                'shots_on_target': home_shots_target,
                'possession': home_possession,
                'pass_accuracy': home_pass_accuracy,
                'fouls': home_fouls,
                'corners': home_corners
            }
            
            away_stats = {
                'shots': away_shots,
                'shots_on_target': away_shots_target,
                'possession': away_possession,
                'pass_accuracy': away_pass_accuracy,
                'fouls': away_fouls,
                'corners': away_corners
            }
            
            if st.sidebar.button("Predict Match Outcome"):
                if st.session_state.model_trained:
                    try:
                        prediction, probabilities = self.predictor.predict_match(
                            home_team, away_team, home_stats, away_stats
                        )
                        
                        st.session_state.prediction = prediction
                        st.session_state.probabilities = probabilities
                        st.session_state.home_stats = home_stats
                        st.session_state.away_stats = away_stats
                        
                    except Exception as e:
                        st.sidebar.error(f"Prediction error: {e}")
                else:
                    st.sidebar.error("Please train the model first!")
    
    def display_data_overview(self):
        """Display data overview"""
        st.header("📊 Data Overview")
        
        if st.session_state.matches_df is not None:
            df = st.session_state.matches_df
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Matches", len(df))
            
            with col2:
                home_wins = len(df[df['outcome'] == 'home_win'])
                st.metric("Home Wins", home_wins)
            
            with col3:
                draw_rate = len(df[df['outcome'] == 'draw']) / len(df)
                st.metric("Draw Rate", f"{draw_rate:.1%}")
            
            # Outcome distribution
            fig = px.pie(
                df, 
                names='outcome', 
                title='Match Outcome Distribution',
                color='outcome',
                color_discrete_map={'home_win': '#2E8B57', 'away_win': '#DC143C', 'draw': '#FFD700'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show sample data
            st.subheader("Sample Match Data")
            st.dataframe(df[['home_team', 'away_team', 'home_goals', 'away_goals', 'outcome']].head(10))
    
    def display_prediction(self):
        """Display prediction results"""
        if hasattr(st.session_state, 'prediction'):
            st.header("🎯 Match Prediction")
            
            prediction = st.session_state.prediction
            probabilities = st.session_state.probabilities
            
            # Create outcome mapping
            outcome_map = {'home_win': 'Home Win', 'away_win': 'Away Win', 'draw': 'Draw'}
            prediction_text = outcome_map[prediction]
            
            # Probability chart
            prob_df = pd.DataFrame({
                'Outcome': ['Home Win', 'Draw', 'Away Win'],
                'Probability': probabilities
            })
            
            fig = px.bar(
                prob_df, 
                x='Outcome', 
                y='Probability',
                color='Outcome',
                color_discrete_map={'Home Win': '#2E8B57', 'Away Win': '#DC143C', 'Draw': '#FFD700'},
                title='Prediction Probabilities'
            )
            fig.update_layout(yaxis_tickformat='.0%')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Prediction: {prediction_text}</h3>
                    <h4>Probabilities:</h4>
                    <p>🏠 Home Win: {probabilities[0]:.1%}</p>
                    <p>🤝 Draw: {probabilities[1]:.1%}</p>
                    <p>✈️ Away Win: {probabilities[2]:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.plotly_chart(fig, use_container_width=True)
            
            # Team statistics comparison
            st.subheader("Team Statistics Comparison")
            
            comparison_data = {
                'Metric': ['Shots', 'Shots on Target', 'Possession', 'Pass Accuracy', 'Fouls', 'Corners'],
                'Home': [
                    st.session_state.home_stats['shots'],
                    st.session_state.home_stats['shots_on_target'],
                    st.session_state.home_stats['possession'],
                    st.session_state.home_stats['pass_accuracy'],
                    st.session_state.home_stats['fouls'],
                    st.session_state.home_stats['corners']
                ],
                'Away': [
                    st.session_state.away_stats['shots'],
                    st.session_state.away_stats['shots_on_target'],
                    st.session_state.away_stats['possession'],
                    st.session_state.away_stats['pass_accuracy'],
                    st.session_state.away_stats['fouls'],
                    st.session_state.away_stats['corners']
                ]
            }
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">⚽ Soccer Match Predictor</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.sidebar_controls()
        
        # Main content
        if st.session_state.matches_df is not None:
            self.display_data_overview()
            
            if st.session_state.model_trained:
                self.display_prediction()
        else:
            st.info("👈 Use the sidebar to fetch match data and train the model!")
            
            # Sample data for demonstration
            st.subheader("How it works:")
            st.markdown("""
            1. **Fetch Data**: Get real match data from API-Football
            2. **Train Model**: Machine learning model learns from historical matches
            3. **Predict**: Input team stats to get match predictions
            4. **Analyze**: View probabilities and team comparisons
            
            **Features used:**
            - Shots and shots on target
            - Ball possession
            - Pass accuracy
            - Fouls and corners
            - Team strength indicators
            """)

# Run the app
if __name__ == "__main__":
    app = SoccerPredictorApp()
    app.run()