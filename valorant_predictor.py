import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

class ValorantMatchPredictor:
    """
    Complete ML Pipeline for Valorant Match Prediction
    Supports multiple models and ensemble methods
    """
    
    def __init__(self, model_type='ensemble'):
        """
        Initialize predictor
        
        Args:
            model_type: 'random_forest', 'xgboost', 'gradient_boost', 'logistic', or 'ensemble'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.elo_ratings = {}  # Store ELO ratings for teams
        
    def calculate_elo_ratings(self, df):
        """
        Calculate ELO ratings for all teams based on match history
        Starting ELO: 1500, K-factor: 32
        """
        elo_ratings = {}
        K = 32  # K-factor
        
        # Sort by date to process chronologically
        df_sorted = df.sort_values('date').copy()
        
        for idx, row in df_sorted.iterrows():
            team1, team2 = row['team1'], row['team2']
            
            # Initialize teams if not seen before
            if team1 not in elo_ratings:
                elo_ratings[team1] = 1500
            if team2 not in elo_ratings:
                elo_ratings[team2] = 1500
            
            # Store current ELO before update
            df_sorted.at[idx, 'team1_elo'] = elo_ratings[team1]
            df_sorted.at[idx, 'team2_elo'] = elo_ratings[team2]
            
            # Calculate expected scores
            expected1 = 1 / (1 + 10 ** ((elo_ratings[team2] - elo_ratings[team1]) / 400))
            expected2 = 1 - expected1
            
            # Actual scores (1 for win, 0 for loss)
            actual1 = 1 if row['winner'] == 1 else 0
            actual2 = 1 - actual1
            
            # Update ELO ratings
            elo_ratings[team1] += K * (actual1 - expected1)
            elo_ratings[team2] += K * (actual2 - expected2)
        
        self.elo_ratings = elo_ratings
        return df_sorted
    
    def engineer_features(self, df):
        """
        Create advanced features from raw data
        """
        df = df.copy()
        
        # Calculate ELO ratings
        df = self.calculate_elo_ratings(df)
        
        # ELO difference (positive means team1 is stronger)
        df['elo_diff'] = df['team1_elo'] - df['team2_elo']
        
        # Win rate difference
        df['winrate_diff'] = df['team1_recent_winrate'] - df['team2_recent_winrate']
        
        # Combined strength score
        df['team1_strength'] = df['team1_elo'] * df['team1_recent_winrate']
        df['team2_strength'] = df['team2_elo'] * df['team2_recent_winrate']
        df['strength_ratio'] = df['team1_strength'] / (df['team2_strength'] + 1e-6)
        
        # Format encoding (Bo3 is standard, others might affect prediction)
        df['is_bo3'] = (df['format'] == 'Bo3').astype(int)
        
        # Momentum features
        df['team1_momentum'] = df['team1_recent_winrate'] * df['team1_elo'] / 1500
        df['team2_momentum'] = df['team2_recent_winrate'] * df['team2_elo'] / 1500
        df['momentum_diff'] = df['team1_momentum'] - df['team2_momentum']
        
        # Underdog indicator (ELO diff > 100 is significant)
        df['is_upset_potential'] = (abs(df['elo_diff']) > 100).astype(int)
        
        # Recent form trend (if winrate > 0.6, team is in good form)
        df['team1_good_form'] = (df['team1_recent_winrate'] >= 0.6).astype(int)
        df['team2_good_form'] = (df['team2_recent_winrate'] >= 0.6).astype(int)
        
        # Both teams in good form (competitive match)
        df['competitive_match'] = (df['team1_good_form'] & df['team2_good_form']).astype(int)
        
        return df
    
    def prepare_features(self, df):
        """
        Select and prepare features for model training
        """
        feature_columns = [
            'team1_elo', 'team2_elo', 'elo_diff',
            'team1_recent_winrate', 'team2_recent_winrate', 'winrate_diff',
            'team1_strength', 'team2_strength', 'strength_ratio',
            'is_bo3', 'team1_momentum', 'team2_momentum', 'momentum_diff',
            'is_upset_potential', 'team1_good_form', 'team2_good_form',
            'competitive_match'
        ]
        
        X = df[feature_columns].copy()
        y = (df['winner'] == 1).astype(int)  # 1 if team1 wins, 0 if team2 wins
        
        return X, y, feature_columns
    
    def build_model(self):
        """
        Build the prediction model based on model_type
        """
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        
        elif self.model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=1
            )
        
        elif self.model_type == 'gradient_boost':
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
        
        elif self.model_type == 'logistic':
            model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        
        elif self.model_type == 'ensemble':
            # Ensemble of multiple models for better accuracy
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
            
            model = VotingClassifier(
                estimators=[('rf', rf), ('xgb', xgb_model), ('gb', gb)],
                voting='soft',
                weights=[1, 2, 1]  # Give more weight to XGBoost
            )
        
        return model
    
    def train(self, df, test_size=0.2):
        """
        Train the model with cross-validation
        """
        print("🔧 Engineering features...")
        df_processed = self.engineer_features(df)
        
        print("📊 Preparing training data...")
        X, y, feature_names = self.prepare_features(df_processed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"🤖 Training {self.model_type} model...")
        self.model = self.build_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        print("✅ Performing cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        # Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Results
        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test)
        }
        
        self.print_results(results)
        return results
    
    def save_model(self, filepath='valorant_model.pkl'):
        """
        Save trained model, scaler, and ELO ratings to file
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'elo_ratings': self.elo_ratings,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='valorant_model.pkl'):
        """
        Load trained model from file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.elo_ratings = model_data['elo_ratings']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data.get('feature_importance', None)
        
        print(f"✅ Model loaded from {filepath}")
        print(f"📊 Model type: {self.model_type}")
        print(f"🎯 ELO ratings for {len(self.elo_ratings)} teams loaded")
    
    def print_results(self, results):
        """
        Print training results in a nice format
        """
        print("\n" + "="*60)
        print("📈 MODEL PERFORMANCE RESULTS")
        print("="*60)
        print(f"Training Accuracy:    {results['train_accuracy']:.4f}")
        print(f"Test Accuracy:        {results['test_accuracy']:.4f}")
        print(f"Cross-Val Accuracy:   {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        print(f"ROC-AUC Score:        {results['roc_auc']:.4f}")
        print("\n📊 Confusion Matrix:")
        print(results['confusion_matrix'])
        print("\n📋 Classification Report:")
        print(results['classification_report'])
        
        if self.feature_importance is not None:
            print("\n🎯 Top 10 Most Important Features:")
            print(self.feature_importance.head(10).to_string(index=False))
        print("="*60 + "\n")
    
    def predict_match(self, team1, team2, team1_winrate, team2_winrate, match_format='Bo3'):
        """
        Predict the outcome of a single match
        
        Returns:
            dict with prediction, probability, and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() or load_model() first.")
        
        # Get or initialize ELO ratings
        team1_elo = self.elo_ratings.get(team1, 1500)
        team2_elo = self.elo_ratings.get(team2, 1500)
        
        # Create feature vector
        match_data = {
            'team1_elo': team1_elo,
            'team2_elo': team2_elo,
            'elo_diff': team1_elo - team2_elo,
            'team1_recent_winrate': team1_winrate,
            'team2_recent_winrate': team2_winrate,
            'winrate_diff': team1_winrate - team2_winrate,
            'team1_strength': team1_elo * team1_winrate,
            'team2_strength': team2_elo * team2_winrate,
            'strength_ratio': (team1_elo * team1_winrate) / (team2_elo * team2_winrate + 1e-6),
            'is_bo3': 1 if match_format == 'Bo3' else 0,
            'team1_momentum': team1_winrate * team1_elo / 1500,
            'team2_momentum': team2_winrate * team2_elo / 1500,
            'momentum_diff': (team1_winrate * team1_elo / 1500) - (team2_winrate * team2_elo / 1500),
            'is_upset_potential': 1 if abs(team1_elo - team2_elo) > 100 else 0,
            'team1_good_form': 1 if team1_winrate >= 0.6 else 0,
            'team2_good_form': 1 if team2_winrate >= 0.6 else 0,
            'competitive_match': 1 if (team1_winrate >= 0.6 and team2_winrate >= 0.6) else 0
        }
        
        X = pd.DataFrame([match_data])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        winner = team1 if prediction == 1 else team2
        confidence = max(probabilities)
        
        result = {
            'winner': winner,
            'team1_win_probability': probabilities[1],
            'team2_win_probability': probabilities[0],
            'confidence': confidence,
            'prediction_strength': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.6 else 'Low',
            'team1_elo': team1_elo,
            'team2_elo': team2_elo
        }
        
        # Print formatted result
        self.print_prediction(team1, team2, result, match_format)
        
        return result
    
    def print_prediction(self, team1, team2, result, match_format):
        """
        Print prediction in a nice format
        """
        print("\n" + "="*60)
        print(f"🎮 MATCH PREDICTION: {team1} vs {team2}")
        print("="*60)
        print(f"Format: {match_format}")
        print(f"\n🏆 Predicted Winner: {result['winner']}")
        print(f"\n📊 Win Probabilities:")
        print(f"  {team1}: {result['team1_win_probability']:.1%}")
        print(f"  {team2}: {result['team2_win_probability']:.1%}")
        print(f"\n💪 Confidence: {result['confidence']:.1%} ({result['prediction_strength']})")
        print(f"\n⭐ ELO Ratings:")
        print(f"  {team1}: {result['team1_elo']:.0f}")
        print(f"  {team2}: {result['team2_elo']:.0f}")
        print("="*60 + "\n")
    
    def get_team_elo(self, team_name):
        """
        Get ELO rating for a specific team
        """
        return self.elo_ratings.get(team_name, 1500)
    
    def list_teams(self):
        """
        List all teams with their ELO ratings
        """
        if not self.elo_ratings:
            print("No teams found. Train the model first.")
            return
        
        teams_df = pd.DataFrame([
            {'Team': team, 'ELO': elo}
            for team, elo in self.elo_ratings.items()
        ]).sort_values('ELO', ascending=False)
        
        print("\n" + "="*60)
        print("🏆 TEAM ELO RANKINGS")
        print("="*60)
        print(teams_df.to_string(index=False))
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('vlr_champions_2025.csv')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    print("🎮 VALORANT MATCH PREDICTOR")
    print(f"📁 Loaded {len(df)} matches\n")
    
    # OPTION 1: Train and save model (run this once)
    print("="*60)
    print("TRAINING NEW MODEL")
    print("="*60)
    
    predictor = ValorantMatchPredictor(model_type='ensemble')
    results = predictor.train(df, test_size=0.2)
    
    # Save the model
    predictor.save_model('valorant_model.pkl')
    
    # Show team rankings
    predictor.list_teams()
    
    # # Example prediction
    # print("\n🔮 Example Prediction:")
    # predictor.predict_match(
    #     team1='Paper Rex',
    #     team2='G2 Esports',
    #     team1_winrate=0.8,
    #     team2_winrate=0.7,
    #     match_format='Bo3'
    # )