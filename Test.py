import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import requests
import Utils

def load_and_plot_results():
    # Directory containing the txt files
    results_dir = 'Results/Points'
    season_points = Utils.get_average_and_highest_points()
    print(season_points)
    
    # Dictionary to store data from each file
    all_points = {}
    
    # Load all txt files
    for filename in os.listdir(results_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(results_dir, filename)
            with open(file_path, 'r') as f:
                points = [float(line.strip()) for line in f.readlines()]
                all_points[filename.replace('.txt', '')] = points
    
    # Create the plot
    plt.figure(figsize=(15, 8))  # Increased figure size for better visibility
    
    # Calculate the number of bars per gameweek
    num_strategies = len(all_points) + 1  # +1 for average only
    bar_width = 0.8 / num_strategies
    
    # Colors for consistent styling
    colors = plt.cm.tab10(np.linspace(0, 1, num_strategies))
    
    # Plot bars for each file and their averages
    for idx, (name, points) in enumerate(all_points.items()):
        # Use the actual length of points array
        gameweeks = range(2, 2 + len(points))
        x = np.array(gameweeks) + (idx * bar_width)
        total_points = sum(points)
        avg_points = np.mean(points)
        
        # Plot bars
        plt.bar(x, points, width=bar_width, label=f"{name} Total: {total_points:.0f}", color=colors[idx])
        
        # Plot average line
        plt.axhline(y=avg_points, color=colors[idx], linestyle='--', alpha=0.5,
                   label=f"{name} Avg: {avg_points:.1f}")
    
    # Add average points bars
    avg_points_data = season_points[season_points['gameweek'] >= 2]['average_points'].values
    x_avg = np.array(range(2, 2 + len(avg_points_data))) + ((len(all_points)) * bar_width)
    total_avg = sum(avg_points_data)
    season_avg = np.mean(avg_points_data)
    
    plt.bar(x_avg, avg_points_data, width=bar_width, label=f"Average Total: {total_avg:.0f}", 
            color='gray', alpha=0.7)
    plt.axhline(y=season_avg, color='gray', linestyle='--', alpha=0.5,
                label=f"Season Avg: {season_avg:.1f}")
    
    plt.xlabel('Gameweek')
    plt.ylabel('Points')
    plt.title('Points per Gameweek Comparison with Season Average')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_player(predictions: pd.DataFrame, name: str, season:str, gw: int):
    # Print specific Player rows for debugging

    specific_rows = predictions[
        (predictions['name'] == name) & 
        (predictions['season'] == season) & 
        (predictions['GW'] == gw)
    ]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    
    # Format the display with specific column widths
    with pd.option_context('display.expand_frame_repr', False):
        print("\nSalah's Predictions for GW 23:")
        print("=" * 100)  # Separator line
        # Display only key columns for better readability
        columns_to_display = ['name', 'team', 'GW', 'opponent_team', 'was_home', 
                            'predicted_next_points', 'confidence_lower', 'confidence_upper', 
                            'next_is_double_gw']
        print(specific_rows[columns_to_display].to_string(index=False))








if __name__ == "__main__":
        # Call the new function to show comparison
    load_and_plot_results()
