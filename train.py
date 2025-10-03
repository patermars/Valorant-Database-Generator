from valorant_predictor import ValorantMatchPredictor
import pandas as pd

# Load data
df = pd.read_csv('/content/vlr_champions_2025.csv')
df['date'] = pd.to_datetime(df['date'])

# Train model
predictor = ValorantMatchPredictor(model_type='ensemble')
results = predictor.train(df, test_size=0.2)

# Save for later use
predictor.save_model('valorant_model.pkl')

# Show team rankings
predictor.list_teams()