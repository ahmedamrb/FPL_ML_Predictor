import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from Trainer import FPLTrainer

class FPLFeatureAnalyzer:
    def __init__(self, trainer: 'FPLTrainer'):
        self.trainer = trainer
        self.feature_importance = {}
        self.feature_correlations = {}
        self.permutation_scores = {}
        
    def analyze_features(self, engineered_data: Dict, n_repeats: int = 5):
        """
        Comprehensive feature analysis for all position models
        """
        for position, data in engineered_data.items():
            print(f"\nAnalyzing features for {position} model...")
            
            df, features = data['df'], data['features']
            model = self.trainer.models[position]
            
            if model is None:
                print(f"No trained model found for {position}")
                continue
                
            # Prepare clean dataset for analysis
            X = df[features]
            y = df['next_gw_points']
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            # Get built-in feature importance
            self.feature_importance[position] = self._get_xgb_importance(model, features)
            
            # Calculate feature correlations
            self.feature_correlations[position] = self._analyze_correlations(X_clean)
            
            # Calculate permutation importance
            self.permutation_scores[position] = self._calculate_permutation_importance(
                model, X_clean, y_clean, features, n_repeats
            )
    
    def _get_xgb_importance(self, model, features: List[str]) -> pd.DataFrame:
        """Get feature importance from XGBoost model"""
        importance_scores = model.feature_importances_
        return pd.DataFrame({
            'feature': features,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
    
    def _analyze_correlations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Analyze feature correlations"""
        return X.corr()
    
    def _calculate_permutation_importance(
        self, model, X: pd.DataFrame, y: pd.Series, 
        features: List[str], n_repeats: int
    ) -> pd.DataFrame:
        """Calculate permutation importance scores"""
        # Use multiple processes for faster computation
        with ProcessPoolExecutor() as executor:
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats, 
                random_state=42, n_jobs=-1
            )
        
        return pd.DataFrame({
            'feature': features,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
    
    def create_feature_importance_plot(self, position: str, top_n: int = 15):
        """Create feature importance plot for a specific position"""
        if position not in self.feature_importance:
            raise KeyError(f"No feature importance data for {position}")
            
        fi_data = self.feature_importance[position].head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=fi_data, x='importance', y='feature', ax=ax)
        ax.set_title(f'Top {top_n} Important Features for {position}')
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        return fig
    
    def create_correlation_heatmap(self, position: str, top_n: int = 15):
        """Create correlation heatmap for top features"""
        if position not in self.feature_correlations:
            raise KeyError(f"No correlation data for {position}")
            
        # Get top features based on importance
        top_features = self.feature_importance[position]['feature'].head(top_n).tolist()
        corr_matrix = self.feature_correlations[position].loc[top_features, top_features]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            ax=ax
        )
        ax.set_title(f'Feature Correlations for {position} (Top {top_n} Features)')
        plt.tight_layout()
        return fig
    
    def display_position_analysis(self, position: str, top_n: int = 15):
        """Display comprehensive analysis for a position"""
        print(f"\n{'='*50}")
        print(f"Analysis for {position} Position")
        print(f"{'='*50}")
        
        # Display top features and their importance scores
        print(f"\nTop {top_n} Important Features:")
        print("-" * 40)
        top_features = self.feature_importance[position].head(top_n)
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        print(top_features.to_string(index=False))
        
        # Create and display plots
        print(f"\nGenerating plots for {position}...")
        fig1 = self.create_feature_importance_plot(position, top_n)
        plt.show()
        
        fig2 = self.create_correlation_heatmap(position, top_n)
        plt.show()
        
        # Close all figures to free memory
        plt.close('all')
    
    def save_analysis(self, output_dir: str):
        """Save all analysis results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for position in self.feature_importance.keys():
            # Save feature importance
            self.feature_importance[position].to_csv(
                output_path / f"{position}_feature_importance.csv"
            )
            
            # Save correlation matrix
            self.feature_correlations[position].to_csv(
                output_path / f"{position}_correlations.csv"
            )
            
            # Save permutation importance
            self.permutation_scores[position].to_csv(
                output_path / f"{position}_permutation_importance.csv"
            )
            
            # Save feature summary
            summary_df = self.get_feature_summary(position)
            if summary_df is not None:
                summary_df.to_csv(output_path / f"{position}_feature_summary.csv")
            
            # Generate and save plots
            fig1 = self.create_feature_importance_plot(position)
            fig1.savefig(output_path / f"{position}_feature_importance.png")
            plt.close(fig1)
            
            fig2 = self.create_correlation_heatmap(position)
            fig2.savefig(output_path / f"{position}_correlation_heatmap.png")
            plt.close(fig2)

    def get_feature_summary(self, position: str, top_n: int = 15) -> pd.DataFrame:
        """
        Generate a comprehensive feature summary including importance metrics
        and correlation statistics
        """
        try:
            stability_df = self.get_feature_stability(position)
            
            # Get correlation statistics
            corr_matrix = self.feature_correlations[position]
            
            # Calculate mean absolute correlation for each feature
            mean_abs_corr = corr_matrix.abs().mean()
            max_abs_corr = corr_matrix.abs().max()
            
            summary_df = stability_df.copy()
            summary_df['mean_abs_correlation'] = summary_df['feature'].map(mean_abs_corr)
            summary_df['max_abs_correlation'] = summary_df['feature'].map(max_abs_corr)
            
            return summary_df.head(top_n)
        except Exception as e:
            print(f"Error generating feature summary for {position}: {str(e)}")
            return None

    def get_feature_stability(self, position: str) -> pd.DataFrame:
        """
        Analyze feature stability by comparing built-in and permutation importance
        """
        if position not in self.feature_importance or position not in self.permutation_scores:
            raise KeyError(f"Missing importance data for {position}")
            
        # Merge different importance metrics
        stability_df = pd.merge(
            self.feature_importance[position],
            self.permutation_scores[position],
            on='feature'
        )
        
        # Calculate stability score
        stability_df['importance_ratio'] = (
            stability_df['importance'] / 
            stability_df['importance_mean']
        )
        
        return stability_df.sort_values('importance', ascending=False)