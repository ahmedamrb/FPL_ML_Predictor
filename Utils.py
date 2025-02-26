import requests
import pandas as pd

def add_actual_points(df):
    latest_season = df['season'].max()
    latest_season_gw = df[df['season'] == latest_season]['GW'].max()
    
    mask = (df['season'] == latest_season) & (df['GW'] == latest_season_gw)
    df['actual_points'] = df['next_gw_points']
    
    if mask.any():
        actual_points_dict = get_actual_points(latest_season_gw +1)
        if actual_points_dict:
            df.loc[mask, 'actual_points'] = df.loc[mask, 'element'].astype(str).map(actual_points_dict)
    
    return df

def get_actual_points(gw):
    try:
        response = requests.get(f"https://fantasy.premierleague.com/api/event/{gw}/live/")
        if response.status_code == 200:
            data = response.json()
            return {str(element['id']): element['stats']['total_points']
                    for element in data['elements']}
        return {}
    except:
        return {}
    
def calculate_confidence_score(pred_value, std_err, historical_std):
    relative_error = std_err / (pred_value + 1)
    relative_hist_std = historical_std / (pred_value + 1)
    return min(max(1 / (1 + relative_error + relative_hist_std), 0), 1)

def get_average_and_highest_points():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch data from FPL API")
    
    data = response.json()
    gameweeks = data["events"]

    gw_points = [
        {
            "gameweek": gw["id"],
            "average_points": gw["average_entry_score"],
            "highest_points": gw["highest_score"]
        }
        for gw in gameweeks if gw["finished"]
    ]

    return pd.DataFrame(gw_points)  # Return as DataFrame
