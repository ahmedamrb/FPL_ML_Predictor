import pandas as pd
import os
import numpy as np
from datetime import datetime
from typing import List, Dict
import glob
from pathlib import Path
from tqdm import tqdm
import requests
import traceback

GW_PATH = Path("Data/Raw/GW")
MERGED_PATH = Path("Data/Merged")
VALIDATION_PATH = Path("Data/Validation")
PROCESSED_PATH = Path("Data/Processed")

class DataMerger:
    

     

    def __init__(self):
        """
        Initialize the CSVMerger class.
        """
        self.merged_df = None

        if not MERGED_PATH.exists(): MERGED_PATH.mkdir(parents=True, exist_ok=True) 
        
    

    def get_gw_files(self) -> List[str]:
        """
        Get all CSV files from the specified Google Drive directory.

        Returns:
            List[str]: List of CSV file paths
        """
        # Use glob to get all CSV files in the directory
        csv_files = glob.glob(os.path.join(GW_PATH, "*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {GW_PATH}")

        print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
        return sorted(csv_files)  # Sort to ensure consistent ordering

    def read_csv_file(self, filepath: str) -> pd.DataFrame:
        """
        Read a single CSV file and add season column.

        Args:
            filepath (str): Path to the CSV file

        Returns:
            pd.DataFrame: DataFrame with added season column
        """
        print(f"Reading file: {os.path.basename(filepath)}")
        df = pd.read_csv(filepath)
        # Extract season name from filename
        season = os.path.splitext(os.path.basename(filepath))[0]
        df['season'] = season
        return df

    def merge_files(self) -> pd.DataFrame:
        """
        Merge all CSV files from the specified directory into a single DataFrame.

        Returns:
            pd.DataFrame: Merged DataFrame
        """
        csv_files = self.get_gw_files()
        dataframes = []

        for filepath in csv_files:
            df = self.read_csv_file(filepath)
            dataframes.append(df)

         # Concatenate all dataframes
        self.merged_df = pd.concat(dataframes, axis=0, ignore_index=True)

        # Rename 'total_points' to 'points'
        self.merged_df = self.merged_df.rename(columns={'total_points': 'points'})
        print(f"Successfully merged {len(csv_files)} files into a single DataFrame")
        return self.merged_df

    def save_merged_data(self, output_path: str) -> None:
        """
        Save the merged DataFrame to a CSV file.

        Args:
            output_path (str): Path to save the merged CSV
        """
        if self.merged_df is not None:
            self.merged_df.to_csv(output_path, index=False)
            print(f"Merged data saved to: {output_path}")
        else:
            raise ValueError("No merged data available. Please run merge_files first.")
        
    def merge_save_validate_data(self):
        # Merge all CSV files from the directory
        merged_df = self.merge_files()

        print("\nData summary:")
        print(f"Total seasons processed: {merged_df['season'].nunique()}")
        print(f"Total entries: {len(merged_df)}")
        print("\nEntries per season:")
        print(merged_df.groupby('season').size())

        # Save merged data
        self.save_merged_data(MERGED_PATH  / f"merged_data.csv")

        # Create validator and generate reports
        validator = DataValidator(merged_df)

        # Save both text and JSON reports
        validator.save_validation_report(VALIDATION_PATH / f"merger_report.txt", format='txt')

    def merge_teams_fixtures(self, fixtures_path: str, teams_path: str) -> pd.DataFrame:
        """
        Process all season data from separate directories for fixtures and teams.

        Args:
            fixtures_path (str): Path to the directory containing fixture files.
            teams_path (str): Path to the directory containing team files.

        Returns:
            pd.DataFrame: Combined DataFrame of all processed seasons.
        """
        print("\nGenerating fixtures DataFrame from raw files...")
        fixture_files = [f for f in os.listdir(fixtures_path) if f.endswith('.csv') and f.startswith('fixtures')]
        team_files = [f for f in os.listdir(teams_path) if f.endswith('.csv') and f.startswith('teams')]

        season_codes = set(f.replace('fixtures', '').replace('.csv', '') for f in fixture_files)
        all_seasons_data = []

        for season_code in tqdm(season_codes, desc="Processing seasons"):
            try:
                fixture_file_path = os.path.join(fixtures_path, f'fixtures{season_code}.csv')
                team_file_path = os.path.join(teams_path, f'teams{season_code}.csv')

                if not os.path.exists(team_file_path):
                    print(f"Team file for season {season_code} not found. Skipping.")
                    continue

                fixtures_df = pd.read_csv(fixture_file_path)
                teams_df = pd.read_csv(team_file_path)

                fixtures_df = fixtures_df.dropna(subset=['kickoff_time'])

                season_df = fixtures_df.merge(
                    teams_df[['id', 'name']],
                    left_on='team_h',
                    right_on='id',
                    how='left'
                ).merge(
                    teams_df[['id', 'name']],
                    left_on='team_a',
                    right_on='id',
                    how='left',
                    suffixes=('_h', '_a')
                )

                result_df = season_df[[
                    'kickoff_time', 'name_h', 'name_a', 'team_h_difficulty', 'team_a_difficulty'
                ]].rename(columns={
                    'name_h': 'team_h_name',
                    'name_a': 'team_a_name'
                })

                result_df['season'] = f'20{season_code.replace("_", "-")}'
                all_seasons_data.append(result_df)
                print(f"Successfully processed season {season_code}.")

            except Exception as e:
                print(f"Error processing season {season_code}: {e}")

        if all_seasons_data:
            combined_df = pd.concat(all_seasons_data, ignore_index=True)
            combined_df = combined_df.sort_values(['kickoff_time'])
            print("Fixtures DataFrame generated successfully.")
            return combined_df
        else:
            print("No data was processed successfully.")
            return pd.DataFrame()
        
class DataValidator:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataValidator class.

        Args:
            df (pd.DataFrame): DataFrame to validate
        """
        self.df = df
        self.validation_report = {}
        # Create output directory if it doesn't exist
        if not VALIDATION_PATH.exists(): VALIDATION_PATH.mkdir(parents=True, exist_ok=True)

    def generate_column_summary(self) -> Dict:
        """
        Generate summary statistics for each column.

        Returns:
            Dict: Dictionary containing column summaries
        """
        column_summary = {}

        for column in self.df.columns:
            summary = {
                'total_rows': len(self.df),
                'missing_values': self.df[column].isna().sum(),
                'missing_percentage': round((self.df[column].isna().sum() / len(self.df)) * 100, 2),
                'unique_values': self.df[column].nunique(),
                'dtype': str(self.df[column].dtype)
            }

            # Add numeric summaries for numeric columns
            # Use pd.api.types.is_numeric_dtype for numeric checks
            # and pd.api.types.is_datetime64_any_dtype for datetime checks
            if pd.api.types.is_numeric_dtype(self.df[column].dtype) or pd.api.types.is_datetime64_any_dtype(self.df[column].dtype):
                summary.update({
                    'min': self.df[column].min(),
                    'max': self.df[column].max(),
                    # Calculate mean and median only for numeric types
                    'mean': round(self.df[column].mean(), 2) if pd.api.types.is_numeric_dtype(self.df[column].dtype) else np.nan,
                    'median': self.df[column].median() if pd.api.types.is_numeric_dtype(self.df[column].dtype) else np.nan,
                })

            column_summary[column] = summary

        return column_summary

    def analyze_season_differences(self) -> Dict:
        """
        Analyze differences between seasons.

        Returns:
            Dict: Dictionary containing season-wise analysis
        """
        season_analysis = {
            'rows_per_season': self.df['season'].value_counts().to_dict(),
            'columns_per_season': {},
            'unique_values_per_season': {}
        }

        for season in sorted(self.df['season'].unique()):
            season_df = self.df[self.df['season'] == season]
            season_analysis['columns_per_season'][season] = list(season_df.columns)

            # Analyze unique values for key columns per season
            key_columns = ['name', 'team', 'position']
            season_unique_values = {}

            for col in key_columns:
                if col in season_df.columns:
                    season_unique_values[col] = season_df[col].nunique()

            season_analysis['unique_values_per_season'][season] = season_unique_values

        return season_analysis

    def generate_text_report(self) -> str:
        """
        Generate a human-readable text validation report.

        Returns:
            str: Formatted text report
        """
        if not self.validation_report:
            self.validation_report = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'column_summary': self.generate_column_summary(),
                'season_analysis': self.analyze_season_differences()
            }

        # Build the text report
        report = []
        report.append("=" * 80)
        report.append("FANTASY PREMIER LEAGUE DATA VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"\nReport Generated: {self.validation_report['timestamp']}")
        report.append(f"Total Rows: {self.validation_report['total_rows']:,}")
        report.append(f"Total Columns: {self.validation_report['total_columns']}")

        # Season Analysis
        report.append("\n" + "=" * 80)
        report.append("SEASON ANALYSIS")
        report.append("=" * 80)

        season_analysis = self.validation_report['season_analysis']
        report.append("\nRows per Season:")
        for season, count in season_analysis['rows_per_season'].items():
            report.append(f"  {season}: {count:,} rows")

        report.append("\nUnique Values per Season:")
        for season, values in season_analysis['unique_values_per_season'].items():
            report.append(f"\n  {season}:")
            for col, count in values.items():
                report.append(f"    {col}: {count:,} unique values")

        # Column Analysis
        report.append("\n" + "=" * 80)
        report.append("COLUMN ANALYSIS")
        report.append("=" * 80)

        for column, summary in self.validation_report['column_summary'].items():
            report.append(f"\nColumn: {column}")
            report.append(f"  Data Type: {summary['dtype']}")
            report.append(f"  Missing Values: {summary['missing_values']:,} ({summary['missing_percentage']}%)")
            report.append(f"  Unique Values: {summary['unique_values']:,}")

            if 'mean' in summary:
                report.append(f"  Numeric Statistics:")
                report.append(f"    Min: {summary['min']:,}")
                report.append(f"    Max: {summary['max']:,}")
                report.append(f"    Mean: {summary['mean']:,}")
                report.append(f"    Median: {summary['median']:,}")

        return "\n".join(report)

    def save_validation_report(self, output_path: str, format: str = 'txt') -> None:
        """
        Save the validation report to a file.

        Args:
            output_path (str): Path to save the validation report
            format (str): Format to save the report ('txt' or 'json')
        """
        if format == 'txt':
            report_text = self.generate_text_report()
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Text validation report saved to: {output_path}")
        else:
            import json
            if not self.validation_report:
                self.generate_text_report()  # This will populate self.validation_report
            with open(output_path, 'w') as f:
                json.dump(self.validation_report, f, indent=4)
            print(f"JSON validation report saved to: {output_path}")

class DataProcessor:
    def __init__(self):
        """
        Initialize the FeatureEngineer class.
        """
        self.merged_data = None
        if not PROCESSED_PATH.exists(): PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """
        Load merged data from the specified directory and preprocess DGW rows.
        """
        print("\nStarting data loading process...")
        merged_data_path = os.path.join(MERGED_PATH, "merged_data.csv")

        print("Loading merged data...")
        if os.path.exists(merged_data_path):
            self.merged_data = pd.read_csv(merged_data_path)
            print(f"Historical data loaded successfully: {self.merged_data.shape[0]:,} rows.")
            # Preprocess DGW rows immediately after loading
            self.merged_data = self.preprocess_double_game_weeks(self.merged_data)
        else:
            raise FileNotFoundError(f"Historical data not found at {merged_data_path}")

    @staticmethod
    def preprocess_double_game_weeks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess DGW rows by aggregating them into single rows per player per GW.

        Args:
            df (pd.DataFrame): Input DataFrame with raw GW data

        Returns:
            pd.DataFrame: DataFrame with DGW rows aggregated
        """
        print("\nPreprocessing double game weeks...")
        result_df = df.copy()

        # Add is_double_gw flag (will be populated by add_double_game_weeks)
        result_df['is_double_gw'] = False

        # Temporarily mark potential DGW rows by checking duplicates
        duplicate_mask = result_df.duplicated(subset=['season', 'name', 'GW'], keep=False)
        if duplicate_mask.sum() > 0:
            print(f"Found {duplicate_mask.sum()} potential DGW rows to aggregate.")

            # Identify DGW groups
            dgw_groups = result_df[duplicate_mask].groupby(['season', 'name', 'GW'])

            # Aggregate numeric columns and handle categorical/context columns
            agg_dict = {
                'points': 'sum', 'minutes': 'sum', 'goals_scored': 'sum', 'assists': 'sum',
                'clean_sheets': 'sum', 'goals_conceded': 'sum', 'bonus': 'sum', 'bps': 'sum',
                'saves': 'sum', 'penalties_saved': 'sum', 'creativity': 'sum', 'influence': 'sum',
                'threat': 'sum', 'ict_index': 'sum', 'own_goals': 'sum', 'penalties_missed': 'sum',
                'red_cards': 'sum', 'yellow_cards': 'sum',
                'xP': 'sum',  # Assuming xP is present; add only if in your data
                'value': 'mean', 'selected': 'mean', 'transfers_balance': 'mean',
                'transfers_in': 'mean', 'transfers_out': 'mean',  # Changed to mean per your request
                'was_home': 'first', 'team': 'first', 'position': 'first', 'kickoff_time': 'first',
                'opponent_team': 'first', 'element': 'first'  # Assuming element is present
            }
            dgw_agg = dgw_groups.agg(agg_dict).reset_index()

            # Add dgw_factor (2 for DGW, 1 for SGW)
            dgw_agg['dgw_factor'] = 2
            dgw_agg['is_double_gw'] = True

            # Keep non-DGW rows
            sgw_df = result_df[~duplicate_mask].copy()
            sgw_df['dgw_factor'] = 1

            # Combine aggregated DGW and SGW rows
            result_df = pd.concat([sgw_df, dgw_agg], ignore_index=True)
            result_df = result_df.sort_values(['season', 'name', 'GW'])

            print(f"Aggregated {len(dgw_agg)} DGW rows. New shape: {result_df.shape}")
        else:
            result_df['dgw_factor'] = 1
            print("No DGW rows detected.")

        return result_df

    @staticmethod
    def add_double_game_weeks(original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced method to flag current and next DGWs accurately using fixture data.

        Args:
            original_df (pd.DataFrame): DataFrame with player data

        Returns:
            pd.DataFrame: DataFrame with is_double_gw and next_is_double_gw columns
        """
        print("\nAdding double game week information...")
        fixtures_path = 'Data/Raw/Fixtures'
        teams_path = 'Data/Raw/Teams'
        
        fixture_files = [f for f in os.listdir(fixtures_path) if f.endswith('.csv') and f.startswith('fixtures')]
        team_files = [f for f in os.listdir(teams_path) if f.endswith('.csv') and f.startswith('teams')]
        season_codes = set(f.replace('fixtures', '').replace('.csv', '') for f in fixture_files)

        # Initialize an empty list to collect DataFrames, avoiding initial column conflicts
        dgw_list = []
        for season_code in season_codes:
            fixture_file_path = os.path.join(fixtures_path, f'fixtures{season_code}.csv')
            team_file_path = os.path.join(teams_path, f'teams{season_code}.csv')

            if not os.path.exists(team_file_path):
                print(f"Team file for season {season_code} not found. Skipping.")
                continue

            fixtures_df = pd.read_csv(fixture_file_path)
            teams_df = pd.read_csv(team_file_path)
            fixtures_df = fixtures_df.dropna(subset=['kickoff_time', 'event'])

            team_id_to_name = dict(zip(teams_df['id'], teams_df['name']))
            fixtures_df['team_h'] = fixtures_df['team_h'].map(team_id_to_name)
            fixtures_df['team_a'] = fixtures_df['team_a'].map(team_id_to_name)

            home_counts = fixtures_df.groupby(['event', 'team_h']).size().reset_index(name='count')
            away_counts = fixtures_df.groupby(['event', 'team_a']).size().reset_index(name='count')
            home_counts.rename(columns={'team_h': 'team'}, inplace=True)
            away_counts.rename(columns={'team_a': 'team'}, inplace=True)
            combined_counts = pd.concat([home_counts, away_counts])
            total_counts = combined_counts.groupby(['event', 'team']).sum().reset_index()

            double_game_weeks = total_counts[total_counts['count'] >= 2]
            season_code = '20' + season_code if len(season_code.split('_')[0]) == 2 else season_code
            double_game_weeks['season'] = season_code
            # Explicitly select and rename columns before appending
            dgw_season = double_game_weeks[['season', 'event', 'team']].rename(columns={'event': 'GW'})
            dgw_list.append(dgw_season)

        # Concatenate all collected DataFrames at once
        if dgw_list:
            d_gw_df = pd.concat(dgw_list, ignore_index=True)
            # Ensure GW is numeric and drop NaN
            d_gw_df['GW'] = pd.to_numeric(d_gw_df['GW'], errors='coerce')
            d_gw_df = d_gw_df.dropna(subset=['GW'])
            d_gw_df['GW'] = d_gw_df['GW'].astype(int)
            d_gw_df = d_gw_df.sort_values(by=['season', 'GW', 'team']).reset_index(drop=True)
        else:
            print("No double game weeks found.")
            d_gw_df = pd.DataFrame(columns=['season', 'GW', 'team'])

        # Debugging: Check column names
        print(f"d_gw_df columns: {list(d_gw_df.columns)}")

        results_df = original_df.copy()
        results_df['next_is_double_gw'] = False

        # Update DGW flags
        for season in results_df['season'].unique():
            season_mask = results_df['season'] == season
            for team in results_df[season_mask]['team'].unique():
                team_mask = results_df['team'] == team
                combined_mask = season_mask & team_mask
                double_gws = d_gw_df[(d_gw_df['season'] == season) & (d_gw_df['team'] == team)]['GW'].values
                results_df.loc[combined_mask & results_df['GW'].isin(double_gws), 'is_double_gw'] = True
                
                # Set next_is_double_gw
                current_gws = results_df[combined_mask]['GW']
                for gw in current_gws:
                    if gw + 1 in double_gws:
                        results_df.loc[combined_mask & (results_df['GW'] == gw), 'next_is_double_gw'] = True

        print("Double game week flags added successfully.")
        return results_df
    
    @staticmethod
    def add_total_points(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cumulative total points, accounting for aggregated DGW rows.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with total_points column added
        """
        print("\nCalculating total points for each player by season...")
        result_df = df.copy()
        result_df = result_df.sort_values(['season', 'name', 'GW'])
        result_df['total_points'] = result_df.groupby(['season', 'name'])['points'].cumsum()
        print("Season Total Points added successfully.")
        return result_df

    def process_data(self):
        """
        Process merged data with new features, including DGW preprocessing.

        Returns:
            processed DataFrame
        """
        print("\nLoading and Preprocessing Merged Data...")
        if self.merged_data is None:
            self.load_data()

        print("\nGenerating fixtures_difficulty DataFrame...")
        from PreProcessing import DataMerger  # Assuming DataMerger is in the same module/file
        fixtures_difficulty_df = DataMerger().merge_teams_fixtures('Data/Raw/fixtures', 'Data/Raw/teams')

        print("\nData summary:")
        print(f"Total seasons processed: {fixtures_difficulty_df['season'].nunique()}")
        print(f"Total matches: {len(fixtures_difficulty_df)}")
        print("\nMatches per season:")
        print(fixtures_difficulty_df.groupby('season').size())

        print("\nProcessing and adding columns...")
        processed_merged_data = (self.merged_data
                               .pipe(self.add_total_points)
                               .pipe(lambda df: df.replace({'position': {'GKP': 'GK'}}))
                               .pipe(self.add_value_efficiency)
                               .pipe(self.add_difficulty, fixtures_difficulty_df)
                               .pipe(self.add_double_game_weeks)
                               .pipe(self.add_next_is_home)
                               .pipe(self.add_next_minutes))

        print("\nData processing completed.")
        print(f"Final shape after preprocessing: {processed_merged_data.shape}")
        print(f"Number of DGW rows: {processed_merged_data['is_double_gw'].sum()}")

        print("\nSaving processed data...")
        processed_merged_output_path = PROCESSED_PATH / "processed_data.csv"
        processed_merged_data.to_csv(processed_merged_output_path, index=False)
        print(f"Processed merged data saved to {processed_merged_output_path}")

        from PreProcessing import DataValidator  # Assuming DataValidator is in the same module/file
        validator = DataValidator(processed_merged_data)
        validator.save_validation_report(VALIDATION_PATH / f"processing_report.txt", format='txt')

        print("Data saved successfully.")
        return processed_merged_data
      
    @staticmethod
    def add_value_efficiency(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add points per million (value efficiency) for each player.

        Args:
            df (pd.DataFrame): Input DataFrame with total_points column

        Returns:
            pd.DataFrame: DataFrame with value_efficiency column added
        """
        print("\nCalculating value efficiency (points per million)...")
        result_df = df.copy()

        # Calculate points per million
        result_df['value_efficiency'] = (result_df['total_points'] / (result_df['value'] / 10)).round(2)

        # Handle division by zero or missing values
        result_df['value_efficiency'] = result_df['value_efficiency'].replace([np.inf, -np.inf], np.nan)

        print("Value efficiency calculated successfully.")
        return result_df
    
    @staticmethod
    def add_difficulty(original_df, fixtures_df):
        """
        Maps current and next opponent difficulty based on team matches in fixtures.

        Args:
            original_df (pd.DataFrame): Original DataFrame with player data
            fixtures_df (pd.DataFrame): DataFrame containing fixture difficulty

        Returns:
            pd.DataFrame: DataFrame with opponent difficulty columns added
        """
        print("\nAdding opponent difficulty information...")

        # Convert kickoff_time to datetime
        original_df['kickoff_time'] = pd.to_datetime(original_df['kickoff_time'])
        fixtures_df['kickoff_time'] = pd.to_datetime(fixtures_df['kickoff_time'])

        # Create dictionaries for quick lookups
        current_difficulty_map = {}
        next_difficulty_map = {}

        print("Processing fixtures to create difficulty mappings...")
        for _, fixture in tqdm(fixtures_df.iterrows(), total=len(fixtures_df), desc="Mapping fixtures"):
            key_home = (fixture['team_h_name'], fixture['kickoff_time'])
            key_away = (fixture['team_a_name'], fixture['kickoff_time'])
            current_difficulty_map[key_home] = fixture['team_h_difficulty']
            current_difficulty_map[key_away] = fixture['team_a_difficulty']

        print("Generating next match difficulties...")
        team_fixtures = {}
        for team in pd.concat([fixtures_df['team_h_name'], fixtures_df['team_a_name']]).unique():
            team_matches = fixtures_df[(fixtures_df['team_h_name'] == team) | (fixtures_df['team_a_name'] == team)]
            team_fixtures[team] = team_matches.sort_values('kickoff_time')

        for team in tqdm(team_fixtures, desc="Processing next fixtures"):
            matches = team_fixtures[team]
            for i in range(len(matches) - 1):
                current_match = matches.iloc[i]
                next_match = matches.iloc[i + 1]
                if team == next_match['team_h_name']:
                    next_diff = next_match['team_h_difficulty']
                else:
                    next_diff = next_match['team_a_difficulty']
                next_difficulty_map[(team, current_match['kickoff_time'])] = next_diff

        original_df['opponent_difficulty'] = original_df.apply(
            lambda row: current_difficulty_map.get((row['team'], row['kickoff_time']), 3), axis=1
        )
        original_df['next_opp_diff'] = original_df.apply(
            lambda row: next_difficulty_map.get((row['team'], row['kickoff_time']), 3), axis=1
        )

        print("Opponent difficulties added successfully.")
        return original_df
    
        
    @staticmethod 
    def add_next_is_home(original_df: pd.DataFrame):
        """
        Add a boolean column indicating if the next game week is a home game.
        For historical seasons, shifts was_home by -1. 
        Uses FPL API for current season.
        
        Args:
            original_df (pd.DataFrame): Input DataFrame with player data
            
        Returns:
            pd.DataFrame: DataFrame with next_is_home column added
        """
        print("\nAdding next home game information...")
        results_df = original_df.copy()
        
        # Initialize next_is_home column
        results_df['next_is_home'] = None
        
        # Get latest season
        latest_season = results_df['season'].max()
        
        # Process historical seasons by shifting was_home
        for season in results_df['season'].unique():
            if season != latest_season:
                season_mask = results_df['season'] == season
                
                # Process each player in the season
                for player in results_df[season_mask]['name'].unique():
                    player_mask = results_df['name'] == player
                    player_season_mask = season_mask & player_mask
                    
                    # Get player's data sorted by GW
                    player_data = results_df[player_season_mask].sort_values('GW')
                    
                    # Shift was_home up by 1 to get next game's home status
                    next_home = player_data['was_home'].shift(-1)
                    
                    # Update next_is_home
                    results_df.loc[player_data.index, 'next_is_home'] = next_home

        # Handle current season using FPL API
        if latest_season == results_df['season'].max():
            print(f"Fetching current season fixtures from FPL API...")
            try:
                # Get bootstrap data
                bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
                bootstrap_data = requests.get(bootstrap_url).json()
                
                # Create team mappings
                teams_data = bootstrap_data['teams']
                team_name_to_id = {team['name']: team['id'] for team in teams_data}
                team_id_to_name = {team['id']: team['name'] for team in teams_data}
                
                # Get fixtures data
                fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
                fixtures_response = requests.get(fixtures_url).json()
                current_fixtures_df = pd.DataFrame(fixtures_response)
                
                if not current_fixtures_df.empty:
                    current_fixtures_df['kickoff_time'] = pd.to_datetime(current_fixtures_df['kickoff_time'])
                    current_fixtures_df['team_h'] = current_fixtures_df['team_h'].map(team_id_to_name)
                    current_fixtures_df['team_a'] = current_fixtures_df['team_a'].map(team_id_to_name)
                    
                    # Process current season data
                    current_mask = results_df['season'] == latest_season
                    for team in results_df[current_mask]['team'].unique():
                        team_fixtures = current_fixtures_df[
                            (current_fixtures_df['team_h'] == team) | 
                            (current_fixtures_df['team_a'] == team)
                        ].sort_values('kickoff_time')
                        
                        team_mask = results_df['team'] == team
                        team_data = results_df[current_mask & team_mask]
                        
                        for idx, row in team_data.iterrows():
                            current_time = pd.to_datetime(row['kickoff_time'])
                            future_fixtures = team_fixtures[team_fixtures['kickoff_time'] > current_time]
                            
                            if not future_fixtures.empty:
                                next_fixture = future_fixtures.iloc[0]
                                results_df.loc[idx, 'next_is_home'] = (next_fixture['team_h'] == team)
                                
            except Exception as e:
                print(f"Error fetching FPL data: {e}")
                traceback.print_exc()
        
        # Convert to boolean and handle missing values
        results_df['next_is_home'] = results_df['next_is_home'].fillna(False)
        results_df['next_is_home'] = results_df['next_is_home'].astype(bool)
        
        print(f"Next home game information added successfully.")
        print(f"Missing values: {results_df['next_is_home'].isna().sum()}")
        
        return results_df
    
    @staticmethod
    def add_next_minutes(original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add next gameweek's minutes for each player.
        For the latest season, fetches all available gameweek data from FPL API.
        
        Args:
            original_df (pd.DataFrame): Input DataFrame with player data
            
        Returns:
            pd.DataFrame: DataFrame with next_minutes column added
        """
        print("\nAdding next minutes information...")
        results_df = original_df.copy()
        
        # Initialize next_minutes column
        results_df['next_minutes'] = None
        
        # Get latest season
        latest_season = results_df['season'].max()
        
        # Process historical data (previous seasons)
        for season in results_df['season'].unique():
            season_mask = results_df['season'] == season
            
            # For each player in each season
            for player in results_df[season_mask]['name'].unique():
                player_mask = results_df['name'] == player
                player_season_mask = season_mask & player_mask
                
                # Get data for this player and season, sorted by GW
                player_data = results_df[player_season_mask].sort_values('GW')
                
                # Shift the minutes column up by 1 to get next game's minutes
                next_mins = player_data['minutes'].shift(-1)
                
                # Update the next_minutes column
                results_df.loc[player_data.index, 'next_minutes'] = next_mins
        
        # Handle latest season with FPL API data
        latest_season_mask = results_df['season'] == latest_season
        
        if latest_season_mask.any():
            print(f"Fetching gameweek minutes from FPL API for {latest_season}...")
            try:
                # Get bootstrap data for player mapping
                bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
                bootstrap_data = requests.get(bootstrap_url).json()
                
                # Create player mapping from web_name to ID
                players_map = {p['web_name']: p['id'] for p in bootstrap_data['elements']}
                
                # Get current gameweek
                current_gw = next(gw['id'] for gw in bootstrap_data['events'] if gw['is_current'])
                
                # Create a dictionary to store all player minutes by gameweek
                all_player_minutes = {}
                
                # Process each player in the latest season
                unique_players = results_df[latest_season_mask]['name'].unique()
                print(f"Processing {len(unique_players)} players from {latest_season}...")
                
                for player_name in tqdm(unique_players, desc="Fetching player data"):
                    if player_name in players_map:
                        player_id = players_map[player_name]
                        
                        # Get player's history data
                        player_url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
                        player_data = requests.get(player_url).json()
                        
                        if 'history' in player_data and player_data['history']:
                            # Extract minutes data per gameweek
                            for gw_data in player_data['history']:
                                gw = gw_data['round']
                                minutes = gw_data['minutes']
                                
                                # Store in dictionary: player_name -> gw -> minutes
                                if player_name not in all_player_minutes:
                                    all_player_minutes[player_name] = {}
                                all_player_minutes[player_name][gw] = minutes
                
                # Update next_minutes for each player's gameweek
                print("Updating next_minutes column...")
                for idx, row in tqdm(results_df[latest_season_mask].iterrows(), 
                                total=results_df[latest_season_mask].shape[0],
                                desc="Updating rows"):
                    player_name = row['name']
                    current_gw = row['GW']
                    next_gw = current_gw + 1
                    
                    if (player_name in all_player_minutes and 
                        next_gw in all_player_minutes[player_name]):
                        results_df.loc[idx, 'next_minutes'] = all_player_minutes[player_name][next_gw]
            
            except Exception as e:
                print(f"Error fetching FPL data: {e}")
                import traceback
                traceback.print_exc()
        
        # Convert next_minutes to numeric and drop NaN values
        results_df['next_minutes'] = pd.to_numeric(results_df['next_minutes'], errors='coerce')
        # Fill NaN values with 0
        results_df['next_minutes'] = results_df['next_minutes'].fillna(0)
        results_df['next_minutes'] = results_df['next_minutes'].astype(int)

        print("Next minutes information added successfully.")
        return results_df


 