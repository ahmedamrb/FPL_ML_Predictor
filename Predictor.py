import pandas as pd
import Utils as utils
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import json
import logging


class FPLPredictor:

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
        self.models_dir = Path("Models")
        self.test_season = '2024_25'
        
    def load_models(self):
        """Load the saved models and standard errors for each position"""
        print("Loading models and standard errors...")
        
        for position in self.models.keys():
            # Load model
            model_path = self.models_dir / f"{position}_model.pkl"
            std_err_path = self.models_dir / f"{position}_std_err.json"
            
            try:
                with open(model_path, 'rb') as f:
                    self.models[position] = pickle.load(f)
                print(f"Loaded {position} model from {model_path}")
                
                with open(std_err_path, 'r') as f:
                    self.std_errs[position] = json.load(f)['std_err']
                print(f"Loaded {position} standard error from {std_err_path}")
                
            except FileNotFoundError as e:
                print(f"Error loading {position} model or standard error: {e}")
                raise
                
        return self.models, self.std_errs

    def predict(self, engineered_data, season=None, models=None, std_errs=None):
        """
        Unified prediction function for both next GW and specific GW.
        Can use provided models or load saved ones.
        """
        # If no models provided, use saved models
        if models is None or std_errs is None:
            if self.models['GK'] is None:  # Models haven't been loaded yet
                self.load_models()
            models = self.models
            std_errs = self.std_errs

        all_predictions = pd.DataFrame()
        
        for position, data in tqdm(engineered_data.items(), desc="Generating predictions"):
            # Filter data for the specified season if provided
            position_df = data['df']
            if season:
                position_df = position_df[position_df['season'] == season]
            position_df = utils.add_actual_points(position_df)  # Add actual points

            # Generate predictions
            features = data['features']
            position_df['predicted_next_points'] = models[position].predict(position_df[features])

            # Calculate confidence metrics
            z_score = 1.234
            std_err = std_errs[position]
            position_df['confidence_lower'] = position_df['predicted_next_points'] - (z_score * std_err)
            position_df['confidence_upper'] = position_df['predicted_next_points'] + (z_score * std_err)

            # Calculate confidence score
            historical_std = (
                position_df
                .groupby('name')['points'].std()
                .fillna(position_df['points'].std())
            )

            position_df['confidence_score'] = position_df.apply(
                lambda row: utils.calculate_confidence_score(
                    row['predicted_next_points'],
                    std_err,
                    historical_std.get(row['name'], position_df['points'].std())
                ),
                axis=1
            )

            all_predictions = pd.concat([all_predictions, position_df])

        return all_predictions

    def predict_test_season(self, engineered_data):
        """Specifically predict for test season data"""
        # Load models if not already loaded
        if self.models['GK'] is None:
            self.load_models()
            
        # Get predictions for all data
        all_predictions = self.predict(engineered_data)
        
        # Filter for test season
        test_predictions = all_predictions[all_predictions['season'] == self.test_season]
        
        return test_predictions

    def get_gw_recommendations(self, predictions_df, season = None, gw = None, n=5):
        """Get top recommendations for each position"""
        
        gw = gw - 1  

        if season is None or gw is None:  #Get Next GW Recommendations
            season = predictions_df['season'].max()
            gw = predictions_df[predictions_df['season'] == season]['GW'].max()
        gw_predictions_df = predictions_df[(predictions_df['season'] == season) & (predictions_df['GW'] == gw)]

        top_predictions = {}
        for position in gw_predictions_df['position'].unique():
            position_df = gw_predictions_df[gw_predictions_df['position'] == position].copy()
            top_n = position_df.nlargest(n, 'predicted_next_points')
            top_predictions[position] = top_n
            #top_predictions[position] = top_n[['name', 'team', 'predicted_next_points',
            #                                'confidence_lower', 'confidence_upper',
            #                                'confidence_score', 'value', 'actual_points']]
        return top_predictions

    def print_recommendations(self, top_predictions):
        """
        Print formatted predictions for each position with position-specific metrics.

        Parameters:
            top_predictions (dict): Dictionary containing predictions for each position
        """
        season = list(top_predictions.values())[0]['season'].values[0]
        gw = list(top_predictions.values())[0]['GW'].values[0]
        print(f"\nTop 5 Predicted Performers by Position for GW {gw+1}, Season {season}:")

        position_specific_columns = {
            'GK': {
                'base_columns': ['name', 'team', 'prediction_range', 'clean_sheets', 'saves', 'actual_points', 'confidence_score', 'price'],
                'display_names': ['Name', 'Team', 'Predicted Points', 'Clean Sheets', 'Saves', 'Actual', 'Confidence', 'Price']
            },
            'DEF': {
                'base_columns': ['name', 'team', 'prediction_range', 'clean_sheets', 'goals_scored', 'assists', 'actual_points', 'confidence_score', 'price'],
                'display_names': ['Name', 'Team', 'Predicted Points', 'Clean Sheets', 'Goals', 'Assists', 'Actual', 'Confidence', 'Price']
            },
            'MID': {
                'base_columns': ['name', 'team', 'prediction_range', 'goals_scored', 'assists', 'ict_index', 'actual_points', 'confidence_score', 'price'],
                'display_names': ['Name', 'Team', 'Predicted Points', 'Goals', 'Assists', 'ICT', 'Actual', 'Confidence', 'Price']
            },
            'FWD': {
                'base_columns': ['name', 'team', 'prediction_range', 'goals_scored', 'assists', 'ict_index', 'actual_points', 'confidence_score', 'price'],
                'display_names': ['Name', 'Team', 'Predicted Points', 'Goals', 'Assists', 'ICT', 'Actual', 'Confidence', 'Price']
            }
        }

        for position, predictions in top_predictions.items():
            print(f"\n{position}:")
            formatted_predictions = predictions.round(3)

            # Add prediction range
            formatted_predictions['prediction_range'] = formatted_predictions.apply(
                lambda x: f"{x['predicted_next_points']:.1f} ({x['confidence_lower']:.1f}-{x['confidence_upper']:.1f})",
                axis=1
            )

            # Calculate price
            formatted_predictions['price'] = formatted_predictions['value'] / 10

            # Get position-specific columns that exist in the dataframe
            available_columns = [col for col in position_specific_columns[position]['base_columns']
                              if col in formatted_predictions.columns]
            display_names = {
                col: position_specific_columns[position]['display_names'][i]
                for i, col in enumerate(position_specific_columns[position]['base_columns'])
                if col in available_columns
            }

            # Create table with available columns
            table = formatted_predictions[available_columns].copy()

            # Format numeric columns
            numeric_columns = table.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != 'price':  # Price already formatted
                    table[col] = table[col].round(2)

            # Print table with custom headers
            print(tabulate(
                table,
                headers=display_names,
                tablefmt='pipe',
                floatfmt='.2f',
                showindex=False
            ))

            # Print position-specific insights
            print(f"\nPosition Insights ({position}):")
            print(f"Average Predicted Points: {formatted_predictions['predicted_next_points'].mean():.2f}")
            print(f"Average Confidence Score: {formatted_predictions['confidence_score'].mean():.2f}")
            
            # Position-specific metrics with safety checks
            if position == 'GK':
                if 'clean_sheets' in formatted_predictions.columns:
                    print(f"Average Clean Sheet Probability: {formatted_predictions['clean_sheets'].mean():.2f}")
            elif position in ['MID', 'FWD']:
                if 'ict_index' in formatted_predictions.columns:
                    print(f"Average ICT Index: {formatted_predictions['ict_index'].mean():.2f}")
            
            print("-" * 80)  # Separator between positions



class TXTReportGenerator:
    """Class to generate reports for a given set of predictions"""
    def __init__(self, predictions_df):
        self.predictions_df = predictions_df
        self.output_path = Path("Results\Reports")

    def generate_gw_report(self, season: str, gw: int):
        """
        Generate a detailed report for a specific gameweek and season.
        
        Args:
            season (str): Season in format "2024_25"
            gw (int): Gameweek number
            
        Returns:
            str: Generated report content
        """
        # Get predictions using OptimizedPredictor
        predictor = FPLPredictor()
   
        top_predictions = predictor.get_gw_recommendations(self.predictions_df, season, gw)
        
        
        # Generate report header
        report = []
        report.append("-" * 80 + "\n")
        report.append(f"\nTop 5 Predicted Performers for GW {gw} Season {season}")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        total_actual_points = 0
        
        # Generate position-specific sections
        for position in ['GK', 'DEF', 'MID', 'FWD']:
    
            position_content = self._generate_position_section(
                position=position,
                predictions=top_predictions[position],
                full_data=self.predictions_df,
                season=season,
                gw=gw
            )
            report.append(position_content)
            total_actual_points += top_predictions[position]['actual_points'].sum()

        # Add gameweek summary
        report.append(f"Predicted Gameweek {gw} Summary:")
        report.append(f"Total Actual Points Across All Positions: {int(total_actual_points)}\n")
        report.append("-" * 80)

        # Join report content
        full_report = "\n".join(report)

        
        # Save report with UTF-8 encoding
        filename = f"gw_{gw}_season_{season}_predictions.txt"
        season_path = self.output_path / season
        season_path.mkdir(parents=True, exist_ok=True)
        with open(season_path / filename, 'w', encoding='utf-8') as f:
            f.write(full_report)

        print(f"Report for GW {gw} Season {season} has been saved to {season_path / filename}")

        return full_report

    def _generate_position_section(self, position: str, predictions: pd.DataFrame, 
                                 full_data: pd.DataFrame, season: str, gw: int) -> str:
        """Generate report section for a specific position"""
        section = []
        section.append(f"{position}:")

        # Format predictions table
        predictions_table = predictions.copy()
        predictions_table['price'] = predictions_table['value'] / 10
        predictions_table['prediction_range'] = predictions_table.apply(
            lambda x: f"{x['predicted_next_points']:.1f} ({x['confidence_lower']:.1f}-{x['confidence_upper']:.1f})",
            axis=1
        )

        # Define columns for each position's main table
        base_columns = ['name', 'team', 'prediction_range', 'actual_points', 
                       'confidence_score', 'price']
        
        # Format and add main predictions table
        table_df = predictions_table[base_columns].copy()
        # Cast actual_points to int
        table_df['actual_points'] = table_df['actual_points'].fillna(0)   #Cast any points of gameweek hasn't been played yet to zero
        table_df['actual_points'] = table_df['actual_points'].astype(int)
        
        section.append(tabulate(
            table_df,
            headers={
            'name': 'Name',
            'team': 'Team',
            'prediction_range': 'Predicted Points',
            'actual_points': 'Actual Points',
            'confidence_score': 'Confidence Score', 
            'price': 'Price'
            },
            tablefmt='pipe',
            floatfmt='.2f',
            showindex=False
        ))
        section.append("")

        # Add position insights
        section.append(f"Position Insights ({position}):")
        section.append(f"- Average Predicted Points: {predictions_table['predicted_next_points'].mean():.2f}")
        section.append(f"- Average Confidence Score: {predictions_table['confidence_score'].mean():.2f}")
        section.append(f"- Total Actual Points: {predictions_table['actual_points'].sum():.2f}")
        section.append(f"- Combined Price: £{predictions_table['price'].sum():.1f}m")
        section.append("")

        # Calculate season statistics
        season_stats = self._calculate_season_stats(
            position=position,
            players=predictions_table['name'].tolist(),
            full_data=full_data,
            season=season,
            up_to_gw=gw - 1
        )
        
        section.append(f"Season Statistics for Selected Players (Up to GW{gw-1}):")
        section.append(tabulate(season_stats, headers='keys', tablefmt='pipe', floatfmt='.1f', showindex=False))
        section.append("\n")

        return "\n".join(section)

    def _calculate_season_stats(self, position: str, players: list, 
                              full_data: pd.DataFrame, season: str, up_to_gw: int) -> pd.DataFrame:
        """Calculate season statistics for selected players up to a specific gameweek"""
        # Filter data
        season_data = full_data[
            (full_data['season'] == season) &
            (full_data['GW'] <= up_to_gw) &
            (full_data['name'].isin(players))
        ]

        # Calculate games played
        games = season_data.groupby('name').size()

        # Initialize stats based on position
        stats = {'Name': players, 'Games': [games.get(player, 0) for player in players]}

        if position == 'GK':
            stats.update({
                'Clean Sheets': self._sum_stat(season_data, 'clean_sheets', players),
                'Total Saves': self._sum_stat(season_data, 'saves', players),
                'Saves/Game': self._avg_stat(season_data, 'saves', players, games)
            })
        elif position == 'DEF':
            stats.update({
                'Clean Sheets': self._sum_stat(season_data, 'clean_sheets', players),
                'Goals': self._sum_stat(season_data, 'goals_scored', players),
                'Assists': self._sum_stat(season_data, 'assists', players)
            })
        else:  # MID and FWD
            stats.update({
                'Goals': self._sum_stat(season_data, 'goals_scored', players),
                'Assists': self._sum_stat(season_data, 'assists', players),
                'Avg ICT': self._avg_stat(season_data, 'ict_index', players, games),
                'Points/Game': self._avg_stat(season_data, 'points', players, games)
            })

        return pd.DataFrame(stats)

    def _sum_stat(self, data: pd.DataFrame, column: str, players: list) -> list:
        """Calculate sum of a statistic for each player"""
        grouped = data.groupby('name')[column].sum()
        return [int(grouped.get(player, 0)) for player in players]

    def _avg_stat(self, data: pd.DataFrame, column: str, players: list, games: pd.Series) -> list:
        """Calculate average of a statistic for each player"""
        grouped = data.groupby('name')[column].sum()
        return [
            round(grouped.get(player, 0) / games.get(player, 1), 1)
            if games.get(player, 0) > 0 else 0.0
            for player in players
        ]
    
    def predict_season(self, season_df, trainer, feature_engineer):
        """
        Predict all gameweeks for a season using expanding window approach.
        
        Args:
            season_df (pd.DataFrame): Season data to predict
            trainer (Trainer): Trained model object
            feature_engineer (FeatureEngineer): Feature engineering object
        
        Returns:
            pd.DataFrame: Original dataframe with predicted_next_points column
        """
        # Create copy of original dataframe
        df = season_df.copy()
        df['predicted_next_points'] = 0.0
        
        # Process each gameweek
        for gw in range(1, 39):
            # Get players for current gameweek
            gw_mask = df['GW'] == gw
            current_players = df[gw_mask]
            
            # Predict for each position
            for position in ['GKP', 'DEF', 'MID', 'FWD']:
                pos_mask = current_players['position'] == position
                pos_players = current_players[pos_mask]
                
                if len(pos_players) > 0:
                    # Get features for these players
                    features = feature_engineer.get_features(pos_players)
                    
                    # Get predictions using appropriate GW model
                    model = trainer.models[position][gw]
                    predictions = model.predict(features)
                    
                    # Update predictions in main dataframe
                    df.loc[(gw_mask) & (df['position'] == position), 
                        'predicted_next_points'] = predictions
        
        return df

    def generate_gw_html_report(self, season: str, gw: int):
        """
        Generate an HTML version of the gameweek report with interactive position selection buttons.
        
        Args:
            season (str): Season in format "2024_25"
            gw (int): Gameweek number
        
        Returns:
            str: Generated HTML report content
        """
        logging.info(f"Generating HTML report for GW {gw} Season {season}")

        # Get predictions using FPLPredictor
        predictor = FPLPredictor()
        top_predictions = predictor.get_gw_recommendations(self.predictions_df, season, gw)

        # Calculate total actual points across all positions
        total_actual_points = sum(df['actual_points'].sum() for df in top_predictions.values())

        # HTML header with FPL styling and JavaScript for interactivity
        html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>FPL GW{gw} Predictions - Season {season}</title>
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
            <style>
                body {{
                    font-family: 'Montserrat', sans-serif;
                    background-color: #38003C; /* FPL Purple */
                    color: #FFFFFF;
                    padding: 20px;
                    margin: 0;
                }}
                h1 {{
                    color: #00FF87; /* FPL Green */
                    text-align: center;
                    margin-bottom: 20px;
                    font-size: 2.5em;
                    font-weight: 700;
                }}
                h2 {{
                    color: #E90052; /* FPL Pink */
                    font-size: 1.8em;
                    margin: 30px 0 15px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background-color: #1E1E1E;
                    font-size: 1.2em;
                }}
                th, td {{
                    padding: 15px;
                    text-align: left;
                    border-bottom: 1px solid #D4B8E9;
                }}
                th {{
                    background-color: #E90052; /* FPL Pink */
                    color: #FFFFFF;
                }}
                tr:nth-child(even) {{
                    background-color: #2A2A2A;
                }}
                .insights, .summary {{
                    background-color: #1E1E1E;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                    border: 2px solid #00FF87;
                }}
                .insights p, .summary p {{
                    margin: 10px 0;
                    font-size: 1.1em;
                }}
                .insights-title, .summary-title {{
                    color: #00FF87;
                    font-size: 1.5em;
                    margin-bottom: 15px;
                    font-weight: 700;
                }}
                .position-section {{
                    display: none; /* Hidden by default */
                }}
                .position-section.active {{
                    display: block; /* Shown when active */
                }}
                .button-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .position-button {{
                    background-color: #E90052; /* FPL Pink */
                    color: #FFFFFF;
                    border: none;
                    padding: 10px 20px;
                    margin: 0 10px;
                    font-size: 1.2em;
                    font-family: 'Montserrat', sans-serif;
                    cursor: pointer;
                    border-radius: 5px;
                    transition: background-color 0.3s;
                }}
                .position-button:hover {{
                    background-color: #00FF87; /* FPL Green */
                }}
                .position-button.active {{
                    background-color: #00FF87; /* FPL Green */
                    font-weight: 700;
                }}
                @media (max-width: 600px) {{
                    table {{
                        font-size: 1em;
                    }}
                    h1 {{
                        font-size: 2em;
                    }}
                    h2 {{
                        font-size: 1.5em;
                    }}
                    .insights, .summary {{
                        padding: 15px;
                    }}
                    .position-button {{
                        padding: 8px 15px;
                        font-size: 1em;
                        margin: 5px;
                    }}
                }}
            </style>
            <script>
                function showPosition(position) {{
                    // Hide all position sections
                    document.querySelectorAll('.position-section').forEach(section => {{
                        section.classList.remove('active');
                    }});
                    // Remove active class from all buttons
                    document.querySelectorAll('.position-button').forEach(button => {{
                        button.classList.remove('active');
                    }});
                    // Show selected position section
                    document.getElementById(position).classList.add('active');
                    // Add active class to clicked button
                    document.getElementById('btn-' + position).classList.add('active');
                }}
                // Show GK by default on page load
                window.onload = function() {{
                    showPosition('GK');
                }};
            </script>
        </head>
        <body>
            <h1>Top 5 Predicted Performers for GW {gw} Season {season}</h1>
            <p style="text-align: center; font-size: 1.1em;">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <!-- Position Selection Buttons -->
            <div class="button-container">
                <button id="btn-GK" class="position-button" onclick="showPosition('GK')">GK</button>
                <button id="btn-DEF" class="position-button" onclick="showPosition('DEF')">DEF</button>
                <button id="btn-MID" class="position-button" onclick="showPosition('MID')">MID</button>
                <button id="btn-FWD" class="position-button" onclick="showPosition('FWD')">FWD</button>
            </div>
        """

        # Generate position sections
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            predictions_table = top_predictions[position].copy()
            predictions_table['price'] = predictions_table['value'] / 10
            predictions_table['prediction_range'] = predictions_table.apply(
                lambda x: f"{x['predicted_next_points']:.1f} ({x['confidence_lower']:.1f}-{x['confidence_upper']:.1f})",
                axis=1
            )
            predictions_table['actual_points'] = predictions_table['actual_points'].fillna(0).astype(int)

            # Wrap each position in a div with a unique ID
            html += f'<div id="{position}" class="position-section">'
            html += f"<h2>{position}:</h2>"

            # Main predictions table
            base_columns = ['web_name', 'team', 'prediction_range', 'actual_points', 'confidence_score', 'price']
            display_names = ['Name', 'Team', 'Predicted Points', 'Actual Points', 'Confidence Score', 'Price']

            html += "<table><thead><tr>"
            for header in display_names:
                html += f"<th>{header}</th>"
            html += "</tr></thead><tbody>"

            for _, row in predictions_table[base_columns].iterrows():
                html += "<tr>"
                for col in base_columns:
                    value = row[col]
                    if col == 'price':
                        value = f"£{value:.1f}m"
                    elif col == 'actual_points':
                        value = int(value)
                    elif col == 'confidence_score':
                        value = f"{value:.2f}"
                    html += f"<td>{value}</td>"
                html += "</tr>"
            html += "</tbody></table>"

            # Position insights
            html += f'<div class="insights"><div class="insights-title">Position Insights ({position}):</div>'
            html += f"<p>- Average Predicted Points: {predictions_table['predicted_next_points'].mean():.2f}</p>"
            html += f"<p>- Average Confidence Score: {predictions_table['confidence_score'].mean():.2f}</p>"
            html += f"<p>- Total Actual Points: {predictions_table['actual_points'].sum():.2f}</p>"
            html += f"<p>- Combined Price: £{predictions_table['price'].sum():.1f}m</p>"

            # Season statistics
            season_stats = self._calculate_season_stats(
                position=position,
                players=predictions_table['name'].tolist(),
                full_data=self.predictions_df,
                season=season,
                up_to_gw=gw - 1
            )

            html += f"<p>Season Statistics for Selected Players (Up to GW{gw-1}):</p>"
            html += "<table><thead><tr>"
            for header in season_stats.columns:
                html += f"<th>{header}</th>"
            html += "</tr></thead><tbody>"

            for _, row in season_stats.iterrows():
                html += "<tr>"
                for col in season_stats.columns:
                    value = row[col]
                    if isinstance(value, float):
                        value = f"{value:.1f}"
                    html += f"<td>{value}</td>"
                html += "</tr>"
            html += "</tbody></table></div>"
            html += "</div>"  # Close position-section div

        # Gameweek summary
        html += '<div class="summary"><div class="summary-title">Gameweek Summary</div>'
        html += f"<p>Predicted Gameweek {gw} Summary:</p>"
        html += f"<p>Total Actual Points Across All Positions: {int(total_actual_points)}</p>"
        html += "</div></body></html>"

        # Save the HTML report
        html_path = self.output_path / "HTML Predictions" / season
        html_path.mkdir(parents=True, exist_ok=True)
        filename = f"gw_{gw}_season_{season}_predictions.html"
        full_path = html_path / filename
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logging.info(f"HTML report saved to {full_path}")
        except Exception as e:
            logging.error(f"Failed to save HTML report: {str(e)}")
            raise

        return html
    