from valorant_predictor import ValorantMatchPredictor

# Load trained model
predictor = ValorantMatchPredictor()
predictor.load_model('valorant_model.pkl')

# Make predictions
predictor.predict_match(
    team1='FNATIC',
    team2='NRG',
    team1_winrate=0.8,
    team2_winrate=1,
    match_format='Bo5'
)