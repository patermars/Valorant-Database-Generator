from valorant_predictor import ValorantMatchPredictor

# Load trained model
predictor = ValorantMatchPredictor()
predictor.load_model('valorant_model.pkl')

# Make predictions
predictor.predict_match(
    team1='FNATIC',
    team2='NRG',
    team1_winrate=0.8,
    team2_winrate=0.8,
    match_format='Bo3'
)

# predictor.predict_match(
#     team1='G2 Esports',
#     team2='Sentinels',
#     team1_winrate=0.75,
#     team2_winrate=0.68,
#     match_format='Bo5'
# )