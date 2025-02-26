from Collectors import DataCollector
from PreProcessing import DataMerger, DataProcessor
from Trainer import FPLTrainer
import pandas as pd
from Predictor import FPLPredictor, TXTReportGenerator
from Manager import FPLManager
from Analyzer import FPLFeatureAnalyzer
import os
import requests
def main():
    # Collecting & updating data from the web
    #data_collector = DataCollector()
    #data_collector.update()
    
    # Merging & Validating the data
    #data_merger = DataMerger()
    #data_merger.merge_save_validate_data()

    # Processing the data
    #data_processor = DataProcessor()
    #data_processor.process_data()



    #Training
    fpl_trainer = FPLTrainer()
    prepared_data = pd.read_csv('Data/Processed/processed_data.csv')

                # Print specific rows for Salah
    salah_rows = prepared_data[
        (prepared_data['name'] == 'Mohamed Salah')
    ]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    
    # Format the display with specific column widths
    with pd.option_context('display.expand_frame_repr', False):
        print("\nSalah's DATA")
        print("=" * 100)  # Separator line
        # Display only key columns for better readability
        columns_to_display = ['name', 'team', 'GW', 'was_home', 'next_is_home', 
                            
                            'next_is_double_gw', 'next_minutes', 'minutes']
        print(salah_rows[columns_to_display].to_string(index=False))


    


    engineered_data = fpl_trainer.engineer_features(prepared_data)

    # Train models
    #models, std_errs = fpl_trainer.train_models(engineered_data)






    # Initialize predictor
    predictor = FPLPredictor()

    #Load models and predict test season
    predictions = predictor.predict(engineered_data, season= "2024_25")
    predictions.rename(columns={'element': 'id'}, inplace=True)
    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    data = response.json()
    elements_df = pd.DataFrame(data['elements'])[['id', 'code', 'web_name', 'team']]
    # Merge with your gw_predictions
    predictions = predictions.merge(elements_df[['id', 'code', 'web_name']], on='id', how='left')





    
    # Apply scaling factor only to rows where next_is_double_gw is True
    scaling_factor = 1.8  # 80% points for second game
    mask = predictions['next_is_double_gw'] == True
            
    # Adjust predictions and confidence intervals for double gameweek players
    predictions.loc[mask, 'predicted_next_points'] *= scaling_factor
    predictions.loc[mask, 'confidence_upper'] *= scaling_factor
    predictions.loc[mask, 'confidence_lower'] *= scaling_factor








    # Get recommendations for specific gameweek
    #recommendations = predictor.get_gw_recommendations(predictions, season='2024_25', gw=22)

    # Print recommendations
    #predictor.print_recommendations(recommendations)

    
    fplmanager = FPLManager()
    actual_points_season = 0

 

    points_per_gw = []
    for gw in range(2, 28):

        b_T, op, actual_points_gw = fplmanager.pick_team(predictions, "2024_25", gw)
        points_per_gw.append(actual_points_gw)
        actual_points_season += actual_points_gw

        # Generating reports for all seasons & gameweeks
        r_generator = TXTReportGenerator(predictions)
        r_generator.generate_gw_report("2024_25", gw)
        r_generator.generate_gw_html_report("2024_25", gw)
    print(f"Total season Actual Points = {actual_points_season}") 
    
    # Save points per gameweek to a file for later comparison
    os.makedirs('Results', exist_ok=True)
    with open('Results/Points/new_results.txt', 'w') as f:
        for points in points_per_gw:
            f.write(f"{points}\n")

    

        

    # Plot using matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.bar(range(2, 28), points_per_gw)
    plt.xlabel('Gameweek')
    plt.ylabel('Points')
    plt.title('Points per Gameweek')
    plt.text(0.5, 0.95, f"Total season Actual Points = {actual_points_season}", 
             horizontalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True, alpha=0.3)
    plt.show()
    
"""
    #ANALYSIS
    analyzer = FPLFeatureAnalyzer(fpl_trainer)

    # Run analysis
    analyzer.analyze_features(engineered_data)

    for position in engineered_data:
        # Get feature summary for a position

        # Generate visualizations
        analyzer.display_position_analysis(position)


        # Save all analysis results
        analyzer.save_analysis('Models/Analysis')

"""
    






    #top_recommendations = predictor.get_gw_recommendations(predictions, "2024_25", 22)
    #predictor.print_recommendations(top_recommendations)
   
  






    # Evaluating the model
    #evaluator = ModelEvaluator(model, data)
    #evaluator.evaluate_model()
    #evaluator.plot_results()

if __name__ == "__main__":
    main()