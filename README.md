# FPL Machine Learning Predictor 🧠⚽

> A powerful machine learning system that analyzes historical Fantasy Premier League (FPL) data to predict player performances and optimize your team selections.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/ahmedamrb/FPL_ML_Predictor/graphs/commit-activity)

## 📋 Overview

This project uses data science and machine learning techniques to analyze historical FPL data, helping you make data-driven decisions for your fantasy team. Whether you're a casual player or a serious competitor, these insights can give you an edge in your mini-leagues!

## ✨ Key Features

- **Smart Predictions**: Machine learning models that forecast player points for upcoming gameweeks
- **Value Analysis**: Identify undervalued players with high potential returns
- **Team Optimization**: Automatically generate optimal team selections within FPL constraints
- **Weekly Predictions Report**: Generate a weekly report with the highest 5 players of each position
- **Weekly Team selection**: Generate a weekly report with the recommended team abiding to the FPL constrains


## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/ahmedamrb/FPL_ML_Predictor.git

# Navigate to the project directory
cd FPL_ML_Predictor

# Install required dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run the main program
python main.py

```

## 📊 Data Sources

This project utilizes data from:
- **Official FPL API**: Current season data and live updates
- **Historical Dataset**: Comprehensive historical data compiled by [Vaastav Anand](https://github.com/vaastav/Fantasy-Premier-League/)

## 🛠️ Technologies

- **Core**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: 
  - Scikit-learn (model training and evaluation)
  - XGBoost (gradient boosting for predictions)
- **Visualization**: Matplotlib, Seaborn (optional)

## 🔮 Future Work & Roadmap

We're continuously working to improve this tool with new features and enhancements. Here's what's coming:

### Short-term
- **Subsitutions**: Automatically subsitute players that hasn't played the predicted GW for better prediction analysis.
- **Transfer Suggestions**: Get intelligent recommendations for transfers based on fixtures and form
- **Performance Analytics**: Visualize player and team performance metrics over time

### Medium-term
- **Chip Strategy Optimizer**: Recommendations for when to use Bench Boost, Triple Captain, and other chips

### Long-term Vision
- **Natural Language Reports**: AI-generated weekly summaries and recommendations in plain English


## 📝 Documentation

For detailed documentation on how the prediction models work and how to customize parameters for your needs, see the [docs folder](docs/) or our [project wiki](https://github.com/ahmedamrb/FPL_ML_Predictor/wiki).

## 👥 Contributing

Contributions are welcome! If you feel there is some issues or need more features then, please feel free to create a PR or create an issue highlighting what is missing and what you would like to be added

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

If you have questions or run into issues:

1. Create an issue on GitHub
2. Submit a pull request with improvements

## 💖 Support Development

If you find this tool valuable for your FPL journey, consider supporting ongoing development:

- Bitcoin (BTC): `bc1q3xe99465px0x82taeg26hqc8c7p0x0lckefcpc`
- Ethereum (ETH): `0x02226a0463797D7081C7E17946Ff3eA9c1abA45C`
- Solana (SOL): `85vdagguAQQW9pPD1T81VS6UvzH6XLDobaVfNHGDDCBR`

Your support helps maintain the project and add new features!

## 📚 Citation

```bibtex
@misc{anand2016fantasypremierleague,
  title = {{FPL Historical Dataset}},
  author = {Anand, Vaastav},
  year = {2022},
  howpublished = {Retrieved from https://github.com/vaastav/Fantasy-Premier-League/}
}
```