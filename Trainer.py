import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

class FPLTrainer:
    def __init__(self):
        self.models = {
            'GK': None,
            'DEF': None,
            'MID': None,
            'FWD': None
        }
        self.std_errs = {
            'GK': None,
            'DEF': None,
            'MID': None,
            'FWD': None
        }
        # Define position features once to avoid repetition
        base_features = {
            'performance': ['bonus', 'bps', 'minutes', 'total_points', 'points'],
            'context': ['opponent_difficulty', 'next_opp_diff', 'was_home', 'value', 'next_is_double_gw',
                       'value_efficiency', 'selected', 'transfers_balance']
        }

        # Build position-specific features dynamically
        self.position_features = {
            'GK': {
                'performance': ['saves', 'clean_sheets', 'goals_conceded', 'penalties_saved'] + base_features['performance'],
                'form_base': ['saves', 'clean_sheets', 'goals_conceded', 'points', 'total_points'],
                'context': base_features['context']
            },
            'DEF': {
                'performance': ['clean_sheets', 'goals_conceded', 'goals_scored', 'assists'] + base_features['performance'],
                'form_base': ['clean_sheets', 'goals_conceded', 'goals_scored', 'assists', 'points', 'total_points'],
                'context': base_features['context']
            },
            'MID': {
                'performance': ['goals_scored', 'assists', 'clean_sheets', 'creativity', 'influence', 'threat', 'ict_index'] + base_features['performance'],
                'form_base': ['goals_scored', 'assists', 'creativity', 'influence', 'threat', 'points', 'total_points'],
                'context': base_features['context']
            },
            'FWD': {
                'performance': ['goals_scored', 'assists', 'creativity', 'influence', 'threat', 'ict_index'] + base_features['performance'],
                'form_base': ['goals_scored', 'assists', 'creativity', 'influence', 'threat', 'points', 'total_points'],
                'context': base_features['context']
            }
        }
        self.models_dir = Path("Models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Define season ranges for training, validation, and testing
        self.initial_training_seasons = ['2019_20', '2020_21', '2021_22', '2022_23']
        self.validation_season = '2023_24'
        self.test_season = '2024_25'
        
        # Track training metrics
        self.training_history = {position: [] for position in self.models.keys()}

    def get_season_gameweeks(self, df, season):
        """Get all gameweeks for a specific season."""
        return sorted(df[df['season'] == season]['GW'].unique())

    def split_validation_data(self, df, season):
        """Split a season's data into first and second half."""
        season_gws = self.get_season_gameweeks(df, season)
        mid_point = len(season_gws) // 2
        first_half_gws = season_gws[:mid_point]
        second_half_gws = season_gws[mid_point:]
        
        first_half = df[(df['season'] == season) & (df['GW'].isin(first_half_gws))]
        second_half = df[(df['season'] == season) & (df['GW'].isin(second_half_gws))]
        
        return first_half, second_half

    def engineer_features(self, df):
        print("Feature Engineering position-specific data...")
        df = df.sort_values(by=['season', 'name', 'GW'])

        engineered_data = {}
        for position in self.position_features:
            position_df = df[df['position'] == position].copy()
            features = self.get_position_features(
                self.add_ewma_features(position_df, position),
                position
            )
            position_df['next_gw_points'] = position_df.groupby(['season', 'name'])['points'].shift(-1)
            #position_df['next_is_home'] = position_df.groupby(['season', 'name'])['was_home'].shift(-1)
            engineered_data[position] = {'df': position_df, 'features': features}

        return engineered_data

    def add_ewma_features(self, df, position):
        print(f"Adding EWMA features for {position}...")
        form_features = self.position_features[position]['form_base']

        for feature in form_features:
            if feature in df.columns:
                for window in [3, 6]:
                    df[f'{feature}_ewm_{window}'] = (
                        df.groupby(['season', 'name'])[feature]
                        .ewm(halflife=window)
                        .mean()
                        .reset_index(level=['season', 'name'], drop=True)
                    )
        return df

    def get_position_features(self, df, position):
        features = []
        feature_types = ['performance', 'context']

        features.extend([f for type_ in feature_types
                        for f in self.position_features[position][type_]
                        if f in df.columns])

        features.extend([f'{feature}_ewm_{window}'
                        for feature in self.position_features[position]['form_base']
                        for window in [3, 6]
                        if f'{feature}_ewm_{window}' in df.columns])

        return features

    def get_training_data(self, df, current_gw, position_data):
        """Get training data up to current gameweek."""
        training_mask = (
            (df['season'].isin(self.initial_training_seasons)) |
            ((df['season'] == self.validation_season) & (df['GW'] <= current_gw))
        )
        return df[training_mask]

    def get_cv_folds(self, df, n_folds=5):
        """Create time-based cross-validation folds."""
        # Sort all season-gameweek combinations
        season_gws = (df[['season', 'GW']]
                    .drop_duplicates()
                    .sort_values(['season', 'GW'])
                    .values.tolist())
        
        # Calculate fold size
        fold_size = len(season_gws) // n_folds
        
        # Create folds
        folds = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else len(season_gws)
            val_season_gws = season_gws[start_idx:end_idx]
            
            # Create masks for this fold
            train_mask = df.apply(lambda x: tuple([x['season'], x['GW']]) < tuple(val_season_gws[0]), axis=1)
            val_mask = df.apply(lambda x: any(tuple([x['season'], x['GW']]) == tuple(sg) for sg in val_season_gws), axis=1)
            
            folds.append((train_mask, val_mask))
        
        return folds

    def train_models(self, engineered_data, n_folds=5):
        """Train models using time-based cross-validation."""
        print("Starting cross-validation training process...")
        
        for position, data in engineered_data.items():
            print(f"\nTraining {position} model with {n_folds}-fold cross-validation...")
            df, features = data['df'], data['features']
            
            # Get CV folds
            cv_folds = self.get_cv_folds(df, n_folds)
            
            # Initialize model with best parameters (you might want to tune these)
            model = XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                random_state=42
            )
            
            # Track fold metrics
            fold_metrics = []
            
            # Cross-validation loop
            for fold_idx, (train_mask, val_mask) in enumerate(tqdm(cv_folds, desc=f"CV folds for {position}")):
                # Prepare training data
                X_train = df[train_mask][features]
                y_train = df[train_mask]['next_gw_points']
                
                # Prepare validation data
                X_val = df[val_mask][features]
                y_val = df[val_mask]['next_gw_points']
                
                # Remove NaN values
                train_valid_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
                X_train, y_train = X_train[train_valid_mask], y_train[train_valid_mask]
                
                val_valid_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
                X_val, y_val = X_val[val_valid_mask], y_val[val_valid_mask]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Validate
                val_pred = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                
                # Store metrics
                fold_metrics.append({
                    'fold': fold_idx + 1,
                    'val_rmse': val_rmse,
                    'train_size': len(X_train),
                    'val_size': len(X_val)
                })
                
                print(f"Fold {fold_idx + 1} - Validation RMSE: {val_rmse:.3f}")
            
            # Store CV results
            self.training_history[position] = fold_metrics
            
            # Final training on all data except test season
            print(f"\nTraining final {position} model on all data...")
            full_train_mask = df['season'] != self.test_season
            X_full = df[full_train_mask][features]
            y_full = df[full_train_mask]['next_gw_points']
            
            valid_mask = ~(X_full.isna().any(axis=1) | y_full.isna())
            X_full, y_full = X_full[valid_mask], y_full[valid_mask]
            
            model.fit(X_full, y_full)
            self.models[position] = model
            self.std_errs[position] = np.std(y_full - model.predict(X_full))
            
            # Save final model and metrics
            self.save_model(position, model)
            self.save_training_history(position)
            
            # Print average CV performance
            avg_rmse = np.mean([m['val_rmse'] for m in fold_metrics])
            std_rmse = np.std([m['val_rmse'] for m in fold_metrics])
            print(f"\n{position} CV Results - Avg RMSE: {avg_rmse:.3f} ± {std_rmse:.3f}")
        
        print("\nCross-validation training complete.")
        return self.models, self.std_errs

    def save_training_history(self, position: str):
        """Save training history with cross-validation metrics for a position"""
        history_path = self.models_dir / f"{position}_cv_training_history.json"
        history_data = {
            'cv_folds': self.training_history[position],
            'avg_rmse': np.mean([m['val_rmse'] for m in self.training_history[position]]),
            'std_rmse': np.std([m['val_rmse'] for m in self.training_history[position]])
        }
        with open(history_path, 'w') as f:
            json.dump(history_data, f)
        print(f"Saved {position} cross-validation history to {history_path}")

    def save_model(self, position: str, model):
        """Save model and metadata for a position"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.models_dir / f"{position}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save standard error
        std_err_path = self.models_dir / f"{position}_std_err.json"
        with open(std_err_path, 'w') as f:
            json.dump({'std_err': float(self.std_errs[position])}, f)
            
        print(f"Saved {position} model to {model_path}")
        print(f"Saved {position} standard error to {std_err_path}")

