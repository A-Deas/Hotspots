import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
FACTOR_LIST = ['OD', 'DR', 'SVI Disability']
DATA_NAMES = ['FIPS'] + [f'{yr} data' for yr in range(2014, 2021)]
FULLY_TRAINED_KALMAN_NAMES = ['FIPS'] + [f'{yr} fully trained kals' for yr in range(2014, 2021)]
KALMAN_NAMES = ['FIPS'] + [f'{yr} kals' for yr in range(2014, 2021)]

def construct_paths(dataset, training_years):
    output_histo_path = f'Images/ToLD/{dataset}/Trained on {training_years}'
    data_path = f'Clean Data/{dataset} rates.csv' 
    fully_trained_kalman_path = f'Kalman Predictions/{dataset} Kalman preds.csv'
    kalman_path = f'Kalman Filter Trained on Less Data/Kalman Predictions ToLD/{dataset} Kalman preds trained on {training_years} years.csv' 
    return output_histo_path, data_path, fully_trained_kalman_path, kalman_path

def load_data(data_path, data_names, full_kalman_path, full_kalman_names, kalman_path, kalman_names):
    data_df = pd.read_csv(data_path, header=0, names=data_names)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].astype(float).clip(lower=0)

    full_kals_df = pd.read_csv(full_kalman_path, header=0, names=full_kalman_names)
    full_kals_df['FIPS'] = full_kals_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    full_kals_df[full_kalman_names[1:]] = full_kals_df[full_kalman_names[1:]].astype(float)

    kals_df = pd.read_csv(kalman_path, header=0, names=kalman_names)
    kals_df['FIPS'] = kals_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    kals_df[kalman_names[1:]] = kals_df[kalman_names[1:]].astype(float)
    return data_df, kals_df, full_kals_df

def calculate_error(data_df, full_kals_df, kals_df, year):
    """ Calculate and return a DataFrame with the absolute errors 
        and find the FIPS code with the maximum error """
    
    err_df = data_df[['FIPS']].copy()
    err_df[f'{year} Fully Trained Absolute Errors'] = abs(full_kals_df[f'{year} fully trained kals'] - data_df[f'{year} data'])
    err_df[f'{year} Kals Absolute Errors'] = abs(kals_df[f'{year} kals'] - data_df[f'{year} data'])

    max_error_index = err_df[f'{year} Kals Absolute Errors'].idxmax()
    max_error_fips = err_df.iloc[max_error_index]['FIPS']
    max_error_value = err_df.iloc[max_error_index][f'{year} Kals Absolute Errors']

    return err_df, max_error_fips, max_error_value

def construct_histogram(err_df, output_histo_path, dataset, training_years, year):
    plt.figure(figsize=(8, 6))
    full_errors = err_df[f'{year} Fully Trained Absolute Errors']
    errors = err_df[f'{year} Kals Absolute Errors']

    plt.hist(full_errors, bins=50, alpha=1, label='Fully Trained Model', edgecolor='black', color='blue')
    plt.hist(errors, bins=50, alpha=0.5, label=f'Model Trained on {training_years} Years', edgecolor='black', color='red')

    if dataset == 'OD':
        size = 15
        if year == 2014:
            tick_positions = np.arange(0, 1, 1)
        else: 
            tick_positions = np.arange(0, 220, 20)
    elif dataset == 'DR':
        size = 15
        if year == 2014:
            tick_positions = np.arange(0, 1, 1)
        else:
            tick_positions = np.arange(0, 520, 40)
    elif dataset.startswith('SVI'):
        size = 11
        if year == 2014:
            tick_positions = np.arange(0, 1, 1)
        else: 
            tick_positions = np.arange(0, 220, 20)

    tick_labels = [str(int(x)) for x in tick_positions]
    plt.xticks(tick_positions, tick_labels) 

    title = f'Absolute Error Comparison for the {dataset} Kalmans in {year}'
    plt.title(title, size=size, weight='bold')
    plt.xlabel('Absolute Error', fontsize=12, weight='bold')
    plt.ylabel('Frequency', fontsize=12, weight='bold')
    plt.legend(loc='upper right')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Display and save the histogram
    #plt.savefig(output_histo_path, bbox_inches=None, pad_inches=0, dpi=300)
    #plt.show()

def main():
    for dataset in FACTOR_LIST:
        for training_years in range(1,5):
            for year in range(2020, 2021):
                output_histo_path, data_path, fully_trained_kalman_path, kalman_path  = construct_paths(dataset, training_years)
                data_df, kals_df, full_kals_df = load_data(data_path, DATA_NAMES, fully_trained_kalman_path, FULLY_TRAINED_KALMAN_NAMES, kalman_path, KALMAN_NAMES)
                err_df, max_error_fips, max_error_value = calculate_error(data_df, full_kals_df, kals_df, year)  # Adjusted to receive max error info
                print(f"{dataset} trained on {training_years}: Max Error = {max_error_value}, FIPS = {max_error_fips}")  # Print max error info
                construct_histogram(err_df, output_histo_path, dataset, training_years, year)
        print()

if __name__ == "__main__":
    main()
