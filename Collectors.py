import requests
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataCollector:
    """
    A class to handle downloading and saving FPL data.
    """

    # Class attributes
    gw_data = {
        "2020_21": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2020-21/gws/merged_gw.csv",
        "2021_22": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/gws/merged_gw.csv",
        "2022_23": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/gws/merged_gw.csv",
        "2023_24": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv",
        "2024_25": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv"
    }

    teams_data = {
        "teams20_21": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2020-21/teams.csv",
        "teams21_22": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/teams.csv",
        "teams22_23": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/teams.csv",
        "teams23_24": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/teams.csv",
        "teams24_25": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/teams.csv",
    }

    fixtures_data = {
        "fixtures20_21": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2020-21/fixtures.csv",
        "fixtures21_22": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/fixtures.csv",
        "fixtures22_23": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/fixtures.csv",
        "fixtures23_24": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/fixtures.csv",
        "fixtures24_25": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/fixtures.csv",
    }

    gw_data_path = Path("Data/Raw/GW")
    teams_data_path = Path("Data/Raw/Teams")
    fixtures_data_path = Path("Data/Raw/Fixtures")
    
    def __init__(self):
        self.gw_data_path.mkdir(parents=True, exist_ok=True)
        self.teams_data_path.mkdir(parents=True, exist_ok=True)
        self.fixtures_data_path.mkdir(parents=True, exist_ok=True)

    def download_files(self, file_dict, base_path):
        """
        Download and save files to the specified base path.
        """
        for file_name, url in file_dict.items():
            output_path = base_path / f"{file_name}.csv"
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx, 5xx)
                with open(output_path, "wb") as file:
                    file.write(response.content)
                logging.info(f"Downloaded and saved: {output_path}")
            except requests.RequestException as e:
                logging.error(f"Failed to download {url}: {e}")

    def update(self):
        """
        Download files for all groups.
        """
        self.download_files(self.gw_data, self.gw_data_path)
        self.download_files(self.teams_data, self.teams_data_path)
        self.download_files(self.fixtures_data, self.fixtures_data_path)
        logging.info("Update Complete ........")

if __name__ == "__main__":
    data_collector = DataCollector()
    data_collector.update()

