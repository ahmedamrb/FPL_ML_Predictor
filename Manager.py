import pandas as pd
from tabulate import tabulate
import os
import Utils
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus
from tqdm import tqdm


class FPLManager:
    def __init__(self):
        self.valid_formations = [
            (3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3), (5, 4, 1), (5, 3, 2)
        ]
        self.budget_limit = 1000  # In 0.1M increments (equivalent to £100M)
        self.box_width = 56  # Fixed width for the formation display
        self.position_requirements = {
            'GK': 2,
            'DEF': 5,
            'MID': 5,
            'FWD': 3
        }

    def select_initial_squad(self, gw_predictions, formation):
        """
        Select initial best squad without budget constraints.
        Returns dictionary of DataFrames by position.
        """
        def_count, mid_count, fwd_count = formation
        squad = {}
        team_counts = {}
        
        # Sort players by predicted points within each position
        position_players = {
            pos: gw_predictions[gw_predictions['position'] == pos].sort_values('predicted_next_points', ascending=False)
            for pos in ['GK', 'DEF', 'MID', 'FWD']
        }
        
        # Select starting XI first
        starting_counts = {
            'GK': 1,
            'DEF': def_count,
            'MID': mid_count,
            'FWD': fwd_count
        }
        
        # Initialize squad with empty DataFrames
        squad = {pos: pd.DataFrame() for pos in ['GK', 'DEF', 'MID', 'FWD']}
        
        # Select starting XI first
        for pos, count in starting_counts.items():
            available = position_players[pos].copy()
            selected = []
            
            for _, player in available.iterrows():
                if len(selected) >= count:
                    break
                    
                team = player['team']
                if team_counts.get(team, 0) < 3:
                    selected.append(player)
                    team_counts[team] = team_counts.get(team, 0) + 1
            
            squad[pos] = pd.concat([squad[pos], pd.DataFrame(selected)])
        
        # Select subs - cheaper players with moderate predictions
        sub_counts = {
            'GK': 1,
            'DEF': max(0, 5 - def_count),
            'MID': max(0, 5 - mid_count),
            'FWD': max(0, 3 - fwd_count)
        }
        
        for pos, count in sub_counts.items():
            if count == 0:
                continue
                
            available = position_players[pos]
            # Filter out already selected players
            available = available[~available['name'].isin(squad[pos]['name'])]
            # Sort by value_efficiency for subs
            available = available.sort_values('value_efficiency', ascending=False)
            
            selected = []
            for _, player in available.iterrows():
                if len(selected) >= count:
                    break
                    
                team = player['team']
                if team_counts.get(team, 0) < 3:
                    selected.append(player)
                    team_counts[team] = team_counts.get(team, 0) + 1
            
            squad[pos] = pd.concat([squad[pos], pd.DataFrame(selected)])
        
        return squad

    def optimize_squad_budget(self, squad, gw_predictions, target_budget):
        """
        Optimize squad to meet budget constraints while maintaining team quality.
        """
        total_cost = sum(df['value'].sum() for df in squad.values())
        if total_cost <= target_budget:
            return squad

        # Try optimizing subs first
        starting_counts = {'GK': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        while total_cost > target_budget:
            # Get all players sorted by value_efficiency
            all_players = []
            for pos, df in squad.items():
                is_sub = pd.Series(range(len(df))) >= starting_counts[pos]
                df_with_sub = df.copy()
                df_with_sub['is_sub'] = is_sub
                all_players.extend(df_with_sub.to_dict('records'))
            
            # Sort by is_sub (True first) and value_efficiency (ascending)
            all_players.sort(key=lambda x: (not x['is_sub'], x['value_efficiency']))
            
            replacement_made = False
            for player in all_players:
                pos = player['position']
                
                # Get replacement candidates
                candidates = gw_predictions[
                    (gw_predictions['position'] == pos) &
                    (gw_predictions['value'] < player['value']) &
                    (~gw_predictions['name'].isin(
                        [p['name'] for p in all_players if p['position'] == pos]
                    ))
                ]
                
                if not candidates.empty:
                    # Get team counts excluding current player
                    team_counts = {}
                    for p in all_players:
                        if p['name'] != player['name']:
                            team_counts[p['team']] = team_counts.get(p['team'], 0) + 1
                    
                    # Filter candidates by team constraint
                    valid_candidates = candidates[
                        candidates['team'].apply(lambda x: team_counts.get(x, 0) < 3)
                    ]
                    
                    if not valid_candidates.empty:
                        # Select best replacement by predicted points
                        replacement = valid_candidates.nlargest(1, 'predicted_next_points').iloc[0]
                        
                        # Update squad
                        squad[pos] = squad[pos][squad[pos]['name'] != player['name']]
                        squad[pos] = pd.concat([squad[pos], pd.DataFrame([replacement])])
                        
                        total_cost = sum(df['value'].sum() for df in squad.values())
                        replacement_made = True
                        break
            
            if not replacement_made:
                raise ValueError("Cannot optimize squad within budget constraints")
        
        return squad

    def pick_team(self, predictions_df, season, predicted_gw):
        """
        Pick team using two-phase selection approach.
        """
        gw = predicted_gw - 1

        # Filter data for the specific season and GW
        gw_predictions = predictions_df[
            (predictions_df['season'] == season) & 
            (predictions_df['GW'] == gw)
        ]
        gw_predictions['actual_points'] = gw_predictions['actual_points'].fillna(0)
        
        # Print diagnostic information
        print(f"\nDiagnostic Information for GW {predicted_gw}:")
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            pos_players = gw_predictions[gw_predictions['position'] == pos]
            print(f"\n{pos} Players Available: {len(pos_players)}")
            if len(pos_players) > 0:
                print("Top 5 by predicted points:")
                top_5 = pos_players.nlargest(5, 'predicted_next_points')
                for _, player in top_5.iterrows():
                    print(f"  {player['name']} ({player['team']}) - Points: {player['predicted_next_points']:.1f}, Price: £{player['value']/10:.1f}m")

        # Try each formation
        best_team = None
        best_points = -1
        best_formation = None
        failed_formations = {}

        for formation in self.valid_formations:
            try:
                print(f"\nTrying formation {formation}...")
                
                # Phase 1: Select initial squad
                squad = self.select_initial_squad(gw_predictions, formation)
                
                # Calculate initial cost
                total_cost = sum(df['value'].sum() for df in squad.values())
                print(f"  Initial squad cost: £{total_cost/10:.1f}m")
                
                # Phase 2: Optimize if over budget
                if total_cost > self.budget_limit:
                    print(f"  Squad over budget, optimizing...")
                    squad = self.optimize_squad_budget(squad, gw_predictions, self.budget_limit)
                    total_cost = sum(df['value'].sum() for df in squad.values())
                    print(f"  Optimized squad cost: £{total_cost/10:.1f}m")
                
                # NEW CODE: Reorder players within each position by predicted points
                for pos in squad:
                    if not squad[pos].empty:
                        squad[pos] = squad[pos].sort_values('predicted_next_points', ascending=False).reset_index(drop=True)
                
                # Calculate total points for starting XI
                def_count, mid_count, fwd_count = formation
                starting_xi_points = (
                    squad['GK'].nlargest(1, 'predicted_next_points')['predicted_next_points'].sum() +
                    squad['DEF'].nlargest(def_count, 'predicted_next_points')['predicted_next_points'].sum() +
                    squad['MID'].nlargest(mid_count, 'predicted_next_points')['predicted_next_points'].sum() +
                    squad['FWD'].nlargest(fwd_count, 'predicted_next_points')['predicted_next_points'].sum()
                )
                
                print(f"  Starting XI points: {starting_xi_points:.1f}")
                
                if starting_xi_points > best_points:
                    best_team = squad
                    best_points = starting_xi_points
                    best_formation = formation
                    print("  Success: New best team found!")

            except Exception as e:
                failed_formations[formation] = str(e)
                print(f"  Failed: {str(e)}")

        if best_team is None:
            print("\nFormation attempt results:")
            for formation, reason in failed_formations.items():
                print(f"Formation {formation}: Failed - {reason}")
            raise ValueError(f"Could not create a valid team for gameweek {gw}. See above diagnostic information.")

        # Select captain
        try:
            captain = max(
                pd.concat(best_team.values()).itertuples(),
                key=lambda x: x.predicted_next_points * x.selected * x.points_ewm_3 * x.points_ewm_6
            )
            captain_name = captain.name
        except Exception as e:
            raise ValueError(f"Error selecting captain: {str(e)}")

        print(f"Captain: {captain_name}")

        try:
            output, actual_points_xi = self._print_squad(best_team, best_formation, captain_name, season, gw)
            self.save_formation_to_file(output, season, gw)
            self.generate_html_report(best_team, best_formation, captain_name, season, gw)
        except Exception as e:
            raise ValueError(f"Error printing squad: {str(e)}")

        return best_team, output, actual_points_xi



    def _print_squad(self, full_squad, formation, captain_name, season, gw):
        """
        Print and format the squad details including formation layout and statistics.
        
        Args:
            full_squad (dict): Dictionary containing player data by position
            formation (tuple): Formation setup (def_count, mid_count, fwd_count)
            captain_name (str): Name of the team captain
            season (str): Season identifier
            gw (int): Gameweek number
        
        Returns:
            list: Formatted output lines
        """
        output = []
        output.append(f"FPLManager predicted picks for gw {gw+1} season {season}:")
        output.append("")
        
        # Initialize tracking variables
        squad_table = []
        predicted_points_xi = 0
        actual_points_xi = 0
        total_cost = 0
        
        # Prepare position counts for starting XI
        def_count, mid_count, fwd_count = formation
        starting_xi_counts = {'GK': 1, 'DEF': def_count, 'MID': mid_count, 'FWD': fwd_count}
        
        # Create squad table
        for position, players in full_squad.items():
            # Reset index to ensure correct row numbering for starting XI check
            players_df = players.reset_index(drop=True)
            for idx, player in players_df.iterrows():
                prediction_range = f"{player['predicted_next_points']:.1f} ({player['confidence_lower']:.1f}-{player['confidence_upper']:.1f})"
                actual_points = int(player.get('actual_points', 0))
                
                # Always add to total cost (including subs)
                total_cost += player['value'] / 10
                
                # Determine if player is in starting XI based on row index
                is_in_starting_xi = idx < starting_xi_counts.get(position, 0)
                if is_in_starting_xi:
                    if player['name'] == captain_name:
                        # Double points for captain
                        predicted_points_xi += player['predicted_next_points'] * 2
                        actual_points *= 2
                    else:
                        predicted_points_xi += player['predicted_next_points']
                    actual_points_xi += actual_points
                
                squad_table.append([
                    player['web_name'],
                    player['team'],
                    prediction_range,
                    actual_points,
                    f"{player['confidence_score']:.2f}",
                    f"{player['value'] / 10:.1f}"
                ])

        # Add squad table to output
        headers = ['Name', 'Team', 'Prediction Range', 'Actual Points', 'Confidence Score', 'Price']
        output.append(tabulate(squad_table, headers=headers, tablefmt='pipe', floatfmt='.2f'))
        output.append(f"\nCaptain: {captain_name}")
        
        def format_row(players_df, count, position_type='players'):
            """Helper function to format each row of the formation"""
            box_width = 14  # Width of each player box
            max_formation_width = 56  # Maximum width of the formation display
            
            # Calculate spacing for centering
            total_width = (box_width * count) + (count - 1)  # Width including separators
            left_margin = (max_formation_width - total_width) // 2
            margin = " " * left_margin
            
            # Create the separator line
            separator = "=" * box_width
            separator_line = margin
            for i in range(count):
                separator_line += separator
                if i < count - 1:
                    separator_line += "="
            
            if position_type in ['top', 'bottom']:
                return [separator_line]
                
            # Create player rows
            names_row = margin
            points_row = margin
            
            # Reset index to ensure proper row selection
            players_df = players_df.reset_index(drop=True)
            for i in range(count):
                player = players_df.iloc[i]
                name = player['web_name'][:10].center(12)
                points = int(player.get('actual_points', 0))
                if player['name'] == captain_name:
                    points = f"{points*2} (C)"
                points = str(points).center(12)
                
                names_row += f"|{name}|"
                points_row += f"|{points}|"
                
                if i < count - 1:
                    names_row += ""
                    points_row += ""
            
            return [names_row, points_row, separator_line]
        
        # Print formation layout
        output.append("\nFormation Layout:")
        
        # Goalkeeper
        gk_df = full_squad['GK'].reset_index(drop=True)
        gk_lines = format_row(gk_df, 1, 'top')
        output.extend(gk_lines)
        gk_lines = format_row(gk_df, 1)
        output.extend(gk_lines)
        
        # Defenders
        def_df = full_squad['DEF'].reset_index(drop=True)
        def_lines = format_row(def_df, def_count, 'top')
        output.extend(def_lines)
        def_lines = format_row(def_df, def_count)
        output.extend(def_lines)
        
        # Midfielders
        mid_df = full_squad['MID'].reset_index(drop=True)
        mid_lines = format_row(mid_df, mid_count, 'top')
        output.extend(mid_lines)
        mid_lines = format_row(mid_df, mid_count)
        output.extend(mid_lines)
        
        # Forwards
        fwd_df = full_squad['FWD'].reset_index(drop=True)
        fwd_lines = format_row(fwd_df, fwd_count, 'top')
        output.extend(fwd_lines)
        fwd_lines = format_row(fwd_df, fwd_count)
        output.extend(fwd_lines)
        fwd_lines = format_row(fwd_df, fwd_count, 'bottom')
        output.extend(fwd_lines)
        
        # Substitutes section
        output.append("\nSubs:")
        
        # Collect all subs
        subs_df = pd.DataFrame()
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            if position in starting_xi_counts:
                players_df = full_squad[position].reset_index(drop=True)
                if len(players_df) > starting_xi_counts[position]:
                    subs_df = pd.concat([subs_df, players_df.iloc[starting_xi_counts[position]:]])
        
        if not subs_df.empty:
            subs_df = subs_df.reset_index(drop=True)
            subs_lines = format_row(subs_df, len(subs_df), 'top')
            output.extend(subs_lines)
            subs_lines = format_row(subs_df, len(subs_df))
            output.extend(subs_lines)
            subs_lines = format_row(subs_df, len(subs_df), 'bottom')
            output.extend(subs_lines)
        
        # Add summary statistics
        output.append(f"\nTotal Team Predicted Points = {predicted_points_xi:.1f}")
        output.append(f"Total Team Actual Points = {int(actual_points_xi)}")
        output.append(f"Total Team Cost = £{total_cost:.1f}m")
        
        # Print and return output
        print("\n".join(output))
        return output, actual_points_xi

    def save_formation_to_file(self, output, season, gw):
        """
        Save the formation output to a text file with UTF-8 encoding.
        
        Args:
            filename (str): Base filename
            output (list): The list of strings containing the formation output
            gw (int): Gameweek number
            season (str): Season identifier
        """
        
        # Create directory if it doesn't exist
        save_path = os.path.join('Results', 'Reports', 'Formations')
        os.makedirs(save_path, exist_ok=True)
        
        # Create filename with gameweek and season
        full_filename = os.path.join(save_path, f'gw{gw+1}_season{season}_formation.txt')
        
        # Save the formation output
        with open(full_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output))


    def generate_html_report(self, full_squad, formation, captain_name, season, gw):
        """
        Generate an HTML report for an FPL squad with FPL branding, enhanced summary, and budget progress bar.
        """
        # Calculate totals
        predicted_points_xi = 0
        actual_points_xi = 0
        total_cost = 0
        starting_xi_counts = {'GK': 1, 'DEF': formation[0], 'MID': formation[1], 'FWD': formation[2]}

        for pos, df in full_squad.items():
            if df.empty or not isinstance(df, pd.DataFrame):
                continue
            df = df.reset_index(drop=True)
            for idx, player in df.iterrows():
                total_cost += player.get('value', 0) / 10
                if idx < starting_xi_counts.get(pos, 0):
                    points = player.get('predicted_next_points', 0)
                    actual = int(player.get('actual_points', 0)) if pd.notna(player.get('actual_points')) else 0
                    if player.get('name', '') == captain_name:
                        predicted_points_xi += points * 2
                        actual_points_xi += actual * 2
                    else:
                        predicted_points_xi += points
                        actual_points_xi += actual

        # Calculate budget percentage
        budget_percentage = (total_cost / 100) * 100  # Assuming £100m total budget

        # Get season points average and highest points
        season_points = Utils.get_average_and_highest_points()
        gw_points = season_points[season_points['gameweek'] == gw +1]
        gw_average_points = gw_points['average_points'].values[0] if not gw_points.empty else 'N/A'
        gw_highest_points = gw_points['highest_points'].values[0] if not gw_points.empty else 'N/A'

        # HTML with FPL branding
        html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>FPL GW{gw+1} Report - Season {season}</title>
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
                .captain {{
                    font-weight: bold;
                    color: #00FF87;
                }}
                .pitch-container {{
                    width: 600px;
                    height: 900px;
                    position: relative;
                    margin: 20px auto;
                }}
                .pitch {{
                    width: 100%;
                    height: 100%;
                    background-image: url('https://i.ibb.co/LzyMzgNs/football-pitch.png');
                    background-size: cover;
                    background-position: center;
                    position: relative;
                    border: 3px solid #FFFFFF;
                    border-radius: 8px;
                }}
                .position {{
                    position: absolute;
                    width: 100%;
                    left: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 10px;
                }}
                .player-card {{
                    width: 110px;
                    text-align: center;
                    position: relative;
                }}
                .player-card img {{
                    width: 110px;
                    height: 110px;
                }}
                .player-card .fallback {{
                    width: 110px;
                    height: 110px;
                    background-color: #FFFFFF;
                    color: #000000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 33px;
                }}
                .player-name {{
                    margin-top: 5px;
                    font-size: 16px;
                    color: #000000;
                    font-weight: bold;
                    text-shadow: -1px -1px 0 #FFF, 1px -1px 0 #FFF, -1px 1px 0 #FFF, 1px 1px 0 #FFF;
                }}
                .player-points {{
                    font-size: 14px;
                    color: #000000;
                    font-weight: bold;
                    text-shadow: -1px -1px 0 #FFF, 1px -1px 0 #FFF, -1px 1px 0 #FFF, 1px 1px 0 #FFF;
                }}
                .captain-badge {{
                    position: absolute;
                    top: -15px;
                    left: -15px;
                    background-color: #E90052;
                    color: white;
                    border-radius: 50%;
                    width: 30px;
                    height: 30px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 16px;
                    font-weight: bold;
                    border: 2px solid #FFFFFF;
                }}
                .subs {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 25px;
                    margin: 20px 0;
                }}
                .sub-player {{
                    text-align: center;
                    width: 110px;
                }}
                .sub-player img {{
                    width: 110px;
                    height: 110px;
                }}
                .summary {{
                    text-align: center;
                    margin: 20px auto;
                    background-color: #1E1E1E;
                    padding: 25px;
                    border-radius: 10px;
                    border: 2px solid #00FF87;
                    font-size: 1.2em;
                    max-width: 600px;
                    box-shadow: 0 0 10px rgba(0, 255, 135, 0.3);
                }}
                .summary-title {{
                    color: #00FF87;
                    font-size: 1.5em;
                    margin-bottom: 20px;
                    font-weight: 700;
                }}
                .summary-item {{
                    margin: 10px 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .summary-label {{
                    color: #E90052;
                    font-weight: 700;
                }}
                .summary-value {{
                    color: #FFFFFF;
                }}
                .progress-bar {{
                    width: 100%;
                    background-color: #2A2A2A;
                    height: 20px;
                    border-radius: 10px;
                    overflow: hidden;
                    margin-top: 10px;
                }}
                .progress {{
                    height: 100%;
                    background-color: #00FF87;
                    width: {budget_percentage}%;
                    transition: width 0.5s ease-in-out;
                }}
                @media (max-width: 600px) {{
                    .pitch-container {{
                        width: 100%;
                        height: 600px;
                    }}
                    .player-card, .sub-player {{
                        width: 85px;
                    }}
                    .player-card img, .player-card .fallback, .sub-player img {{
                        width: 85px;
                        height: 85px;
                    }}
                    .player-name {{
                        font-size: 12px;
                    }}
                    .player-points {{
                        font-size: 10px;
                    }}
                    table {{
                        font-size: 1em;
                    }}
                    .summary {{
                        padding: 15px;
                    }}
                }}
            </style>
        </head>
        <body>
            <h1>AI FPLManager Picks for GW {gw+1} Season {season}</h1>
            
            <!-- Enhanced Summary -->
            <div class="summary">
                <div class="summary-title">Team Summary</div>
                <div class="summary-item">
                    <span class="summary-label">Predicted Points:</span>
                    <span class="summary-value">{predicted_points_xi:.1f}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Actual Points:</span>
                    <span class="summary-value">{actual_points_xi}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">GW Average Points:</span>
                    <span class="summary-value">{gw_average_points}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">GW Highest Points:</span>
                    <span class="summary-value">{gw_highest_points}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Budget Used (£{total_cost:.1f}m / £100m):</span>
                    <span class="summary-value">{budget_percentage:.1f}%</span>
                </div>                
                <div class="progress-bar">
                    <div class="progress"></div>
                </div>
                                <div class="summary-item">
                    <span class="summary-label">Captain:</span>
                    <span class="summary-value">{captain_name}</span>
                </div>
            </div>
            
            <!-- Pitch Visualization -->
            <div class="pitch-container">
                <div class="pitch">
        """

        # Position divs
        def_count, mid_count, fwd_count = formation
        position_styles = {
            'GK': 'top: 80%;',
            'DEF': 'top: 60%;',
            'MID': 'top: 40%;',
            'FWD': 'top: 15%;'
        }

        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            if pos not in full_squad or full_squad[pos].empty:
                continue
            count = 1 if pos == 'GK' else formation[0] if pos == 'DEF' else formation[1] if pos == 'MID' else formation[2]
            df = full_squad[pos].head(count)
            
            html += f"""
                <div class="position" style="{position_styles[pos]}">
            """
            
            for _, player in df.iterrows():
                code = str(player.get('code', ''))
                name = player.get('web_name', 'Unknown')
                actual_points = int(player.get('actual_points', 0)) if pd.notna(player.get('actual_points')) else 0
                is_captain = player.get('name', '') == captain_name
                if is_captain:
                    actual_points *= 2

                html += f"""
                    <div class="player-card">
                """
                if code:
                    html += f"""
                        <img src="https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png" 
                            alt="{name}" 
                            onerror="this.style.display='none';this.nextElementSibling.style.display='flex';">
                        <div class="fallback" style="display:none;">{name[0]}</div>
                    """
                else:
                    html += f"""
                        <div class="fallback">{name[0]}</div>
                    """
                
                if is_captain:
                    html += '<div class="captain-badge">C</div>'
                    
                html += f"""
                        <div class="player-name">{name}</div>
                        <div class="player-points">{actual_points}</div>
                    </div>
                """
            
            html += """
                </div>
            """

        html += """
                </div>
            </div>
            
            <!-- Substitutes -->
            <div class="subs">
        """

        # Substitutes
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            if pos not in full_squad or full_squad[pos].empty:
                continue
            df = full_squad[pos].iloc[starting_xi_counts[pos]:]
            for _, player in df.iterrows():
                code = str(player.get('code', ''))
                name = player.get('web_name', 'Unknown')
                actual_points = int(player.get('actual_points', 0)) if pd.notna(player.get('actual_points')) else 0
                
                html += f"""
                    <div class="sub-player">
                """
                if code:
                    html += f"""
                        <img src="https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png" 
                            alt="{name}" 
                            onerror="this.style.display='none';this.nextElementSibling.style.display='flex';">
                        <div class="fallback" style="display:none;">{name[0]}</div>
                    """
                else:
                    html += f"""
                        <div class="fallback">{name[0]}</div>
                    """
                html += f"""
                        <div class="player-name">{name}</div>
                        <div class="player-points">{actual_points}</div>
                    </div>
                """

        html += """
            </div>
            
            <!-- Squad Table -->
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Position</th>
                        <th>Team</th>
                        <th>Predicted Points</th>
                        <th>Actual Points</th>
                        <th>Price</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Squad table
        for pos, df in full_squad.items():
            if df.empty or not isinstance(df, pd.DataFrame):
                continue
            for _, player in df.iterrows():
                name = player.get('name', 'Unknown')
                points = f"{player.get('predicted_next_points', 0):.1f}"
                actual = int(player.get('actual_points', 0)) if pd.notna(player.get('actual_points')) else 0
                price = f"{player.get('value', 0) / 10:.1f}"
                captain_class = 'captain' if name == captain_name else ''
                html += f"""
                    <tr class="{captain_class}">
                        <td>{name}{' (C)' if name == captain_name else ''}</td>
                        <td>{pos}</td>
                        <td>{player.get('team', 'N/A')}</td>
                        <td>{points}</td>
                        <td>{actual}</td>
                        <td>£{price}m</td>
                    </tr>
                """

        html += """
                </tbody>
            </table>
        </body>
        </html>
        """

        # Save the HTML file
        save_path = os.path.join('Results', 'Reports', 'HTML_Reports')
        os.makedirs(save_path, exist_ok=True)
        filename = f'gw{gw+1}_season{season}_report.html'
        try:
            with open(os.path.join(save_path, filename), 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"HTML report saved to {os.path.join(save_path, filename)}")
        except Exception as e:
            raise ValueError(f"Failed to save HTML report: {str(e)}")

 