import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
FACTOR_LIST = ['OD', 'DR', 'SVI Disability']
DATA_NAMES = ['FIPS'] + [f'{yr} data' for yr in range(2014, 2021)]
FULLY_TRAINED_KALMAN_NAMES = ['FIPS'] + [f'{yr} fully trained kals' for yr in range(2014, 2021)]

def construct_paths(dataset, init_year):
    output_histo_path = f'Images/PoLD/{dataset}/initialized_at_{init_year}.png'
    data_path = f'Clean Data/{dataset} rates.csv' 
    fully_trained_kalman_path = f'Kalman Predictions/{dataset} Kalman preds.csv'
    pold_kalman_path = f'KF PoLD/PolD Predictions/{dataset}/{dataset}_initialized_at_{init_year}.csv' 
    return output_histo_path, data_path, fully_trained_kalman_path, pold_kalman_path

def load_data(data_path, data_names, full_kalman_path, full_kalman_names, pold_kalman_path, init_year):
    data_df = pd.read_csv(data_path, header=0, names=data_names)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].astype(float).clip(lower=0)

    full_kals_df = pd.read_csv(full_kalman_path, header=0, names=full_kalman_names)
    full_kals_df['FIPS'] = full_kals_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    full_kals_df[full_kalman_names[1:]] = full_kals_df[full_kalman_names[1:]].astype(float)

    pold_kalman_names = ['FIPS'] + [f'{yr} pold kals' for yr in range(init_year, 2021)]
    pold_kals_df = pd.read_csv(pold_kalman_path, header=0, names=pold_kalman_names)
    pold_kals_df['FIPS'] = pold_kals_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    pold_kals_df[pold_kalman_names[1:]] = pold_kals_df[pold_kalman_names[1:]].astype(float)
    return data_df, full_kals_df, pold_kals_df

def calculate_error(data_df, full_kals_df, pold_kals_df, year):
    """ Calculate and return a DataFrame with the absolute errors 
        and find the FIPS code with the maximum error """
    
    err_df = data_df[['FIPS']].copy()
    err_df[f'{year} Fully Trained Absolute Errors'] = abs(full_kals_df[f'{year} fully trained kals'] - data_df[f'{year} data'])
    err_df[f'{year} PolD Absolute Errors'] = abs(pold_kals_df[f'{year} pold kals'] - data_df[f'{year} data'])
    return err_df

def construct_histogram(err_df, output_histo_path, dataset, init_year, year):
    plt.figure(figsize=(8, 6))
    full_errors = err_df[f'{year} Fully Trained Absolute Errors']
    pold_errors = err_df[f'{year} PolD Absolute Errors']

    full_max_error = np.round(full_errors.max(), 2) if not np.isnan(full_errors.max()) else 0
    pold_max_error = np.round(pold_errors.max(), 2) if not np.isnan(pold_errors.max()) else 0
    overall_max_error = max(full_max_error, pold_max_error)

    # Plot both histograms
    plt.hist(full_errors, bins=50, alpha=0.7, label='Fully Trained Model', edgecolor='black', color='blue')
    n, bins, patches = plt.hist(pold_errors, bins=50, alpha=0.5, label=f'Model Initialized at {init_year}', edgecolor='black', color='darkorange')

    # Determine tick spacing dynamically
    num_ticks = 10  # Adjust this to control the number of ticks
    if dataset == 'DR':
        tick_positions = np.linspace(0, 280, num=num_ticks)
    elif dataset == 'OD':
        tick_positions = np.linspace(0, 90, num=num_ticks)
    elif dataset == 'SVI Disability':
        tick_positions = np.linspace(0, 35, num=num_ticks)

    tick_labels = [str(int(x)) for x in tick_positions]
    plt.xticks(tick_positions, tick_labels) 

    title = f'{year} {dataset} Absolute Error Comparison for the Fully Trained Model vs the Model inialized at {init_year}'
    plt.title(title, size=12, weight='bold')

    plt.xlabel('Absolute Error', fontsize=12, weight='bold')
    plt.ylabel('Frequency', fontsize=12, weight='bold')
    plt.legend(loc='upper right')
    
# Annotate Fully Trained Histogram with an arrow displaying max error
    full_max_error_bin_index = np.digitize([full_max_error], bins) - 1
    full_max_error_bin_index = min(full_max_error_bin_index[0], len(n) - 1)
    full_max_error_bin_height = n[full_max_error_bin_index]

    # Annotate pold Histogram with an arrow displaying max error
    pold_max_error_bin_index = np.digitize([pold_max_error], bins) - 1
    pold_max_error_bin_index = min(pold_max_error_bin_index[0], len(n) - 1)
    pold_max_error_bin_height = n[pold_max_error_bin_index]

    # Set base vertical position
    full_annotate_y = 150
    pold_annotate_y = 150

    # If they are too close, adjust the text position for better spacing
    if abs(full_max_error - pold_max_error) < 10:
        full_annotate_y += 30
        pold_annotate_y -= 30

    plt.annotate(f'{full_max_error}', 
                xy=(full_max_error, full_max_error_bin_height), 
                xytext=(full_max_error, full_annotate_y),  
                ha='center', 
                va='bottom', 
                arrowprops=dict(facecolor='blue', shrink=0.05),
                fontsize=10, weight='bold', color='blue')

    plt.annotate(f'{pold_max_error}', 
                xy=(pold_max_error, pold_max_error_bin_height), 
                xytext=(pold_max_error, pold_annotate_y),  
                ha='center', 
                va='bottom', 
                arrowprops=dict(facecolor='darkorange', shrink=0.05),
                fontsize=10, weight='bold', color='darkorange')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Display and save the histogram
    plt.savefig(output_histo_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    #plt.show()

def main():
    for dataset in FACTOR_LIST:
        for init_year in range(2014, 2020):
            for year in range(2020, 2021):
                output_histo_path, data_path, fully_trained_kalman_path, kalman_path  = construct_paths(dataset, init_year)
                data_df, full_kals_df, pold_kals_df = load_data(data_path, DATA_NAMES, fully_trained_kalman_path, FULLY_TRAINED_KALMAN_NAMES, kalman_path, init_year)
                err_df = calculate_error(data_df, full_kals_df, pold_kals_df, year)  # Adjusted to receive max error info
                construct_histogram(err_df, output_histo_path, dataset, init_year, year)

if __name__ == "__main__":
    main()
