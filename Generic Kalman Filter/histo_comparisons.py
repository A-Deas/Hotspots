import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
FACTOR_LIST = ['OD', 'DR', 'SVI Disability']
DATA_NAMES = ['FIPS'] + [f'{yr} data' for yr in range(2014, 2021)]
SPATIAL_KALMAN_NAMES = ['FIPS'] + [f'{yr} spatial kals' for yr in range(2014, 2021)]
GENERIC_KALMAN_NAMES = ['FIPS'] + [f'{yr} generic kals' for yr in range(2014, 2021)]

def construct_paths(dataset, year):
    output_histo_path = f'Generic Kalman Filter/Histogram Comparisons/{dataset}/{year}_{dataset}_err_histo_comp.png'
    data_path = f'Clean Data/{dataset} rates.csv' 
    spatial_kalman_path = f'Kalman Predictions/{dataset} Kalman preds.csv'
    generic_kalman_path = f'Generic Kalman Filter/Generic Predictions/{dataset} Generic Preds.csv' 
    return output_histo_path, data_path, spatial_kalman_path, generic_kalman_path

def load_data(data_path, data_names, spatial_kalman_path, spatial_kalman_names, generic_kalman_path, generic_kalman_names):
    data_df = pd.read_csv(data_path, header=0, names=data_names)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].astype(float).clip(lower=0)

    spatial_kals_df = pd.read_csv(spatial_kalman_path, header=0, names=spatial_kalman_names)
    spatial_kals_df['FIPS'] = spatial_kals_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    spatial_kals_df[spatial_kalman_names[1:]] = spatial_kals_df[spatial_kalman_names[1:]].astype(float)

    generic_kals_df = pd.read_csv(generic_kalman_path, header=0, names=generic_kalman_names)
    generic_kals_df['FIPS'] = generic_kals_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    generic_kals_df[generic_kalman_names[1:]] = generic_kals_df[generic_kalman_names[1:]].astype(float)
    return data_df, spatial_kals_df, generic_kals_df

def calculate_error(data_df, spatial_kals_df, generic_kals_df, year):
    """ Calculate and return a DataFrame with the absolute errors 
        and find the FIPS code with the maximum error """
    
    err_df = data_df[['FIPS']].copy()
    err_df[f'{year} Spatial Kalman Absolute Errors'] = abs(spatial_kals_df[f'{year} spatial kals'] - data_df[f'{year} data'])
    err_df[f'{year} Generic Kalman Absolute Errors'] = abs(generic_kals_df[f'{year} generic kals'] - data_df[f'{year} data'])
    return err_df

def construct_histogram(err_df, output_histo_path, dataset, year):
    plt.figure(figsize=(8, 6))
    spatial_errors = err_df[f'{year} Spatial Kalman Absolute Errors']
    generic_errors = err_df[f'{year} Generic Kalman Absolute Errors']

    spatial_max_error = np.round(spatial_errors.max(), 2) if not np.isnan(spatial_errors.max()) else 0
    generic_max_error = np.round(generic_errors.max(), 2) if not np.isnan(generic_errors.max()) else 0
    overall_max_error = max(spatial_max_error, generic_max_error)

    # Plot both histograms
    n, bins, patches = plt.hist(spatial_errors, bins=50, alpha=0.7, label='Spatial Kalman Filter', edgecolor='black', color='purple')
    n, bins, patches = plt.hist(generic_errors, bins=50, alpha=0.5, label='Generic Kalman Filter', edgecolor='black', color='darkorange')

    # Determine tick spacing dynamically
    num_ticks = 10  # Adjust this to control the number of ticks
    tick_positions = np.linspace(0, overall_max_error, num=num_ticks)

    tick_labels = [str(int(x)) for x in tick_positions]
    plt.xticks(tick_positions, tick_labels) 

    title = f'{year} {dataset} Absolute Error Comparison of Spatial VS Generic Kalman Filters'
    plt.title(title, size=12, weight='bold')

    plt.xlabel('Absolute Error', fontsize=12, weight='bold')
    plt.ylabel('Frequency', fontsize=12, weight='bold')
    plt.legend(loc='upper right')

    # Annotate Spatial Histogram with an arrow displaying max error
    spatial_max_error_bin_index = np.digitize([spatial_max_error], bins) - 1
    spatial_max_error_bin_index = min(spatial_max_error_bin_index[0], len(n) - 1)
    spatial_max_error_bin_height = n[spatial_max_error_bin_index]

    # Annotate Generic Histogram with an arrow displaying max error
    generic_max_error_bin_index = np.digitize([generic_max_error], bins) - 1
    generic_max_error_bin_index = min(generic_max_error_bin_index[0], len(n) - 1)
    generic_max_error_bin_height = n[generic_max_error_bin_index]

    # Set base vertical position
    spatial_annotate_y = 150
    generic_annotate_y = 150

    # If they are too close, adjust the text position for better spacing
    if abs(spatial_max_error - generic_max_error) < 10:
        spatial_annotate_y += 30
        generic_annotate_y -= 30

    plt.annotate(f'{spatial_max_error}', 
                xy=(spatial_max_error, spatial_max_error_bin_height), 
                xytext=(spatial_max_error, spatial_annotate_y),  
                ha='center', 
                va='bottom', 
                arrowprops=dict(facecolor='purple', shrink=0.05),
                fontsize=10, weight='bold', color='purple')

    plt.annotate(f'{generic_max_error}', 
                xy=(generic_max_error, generic_max_error_bin_height), 
                xytext=(generic_max_error, generic_annotate_y),  
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

def print_prediction_ranges(spatial_kals_df, generic_kals_df, dataset, year):
    """ Print the min and max predictions for both Kalman filters """
    
    spatial_min, spatial_max = spatial_kals_df[f'{year} spatial kals'].min(), spatial_kals_df[f'{year} spatial kals'].max()
    generic_min, generic_max = generic_kals_df[f'{year} generic kals'].min(), generic_kals_df[f'{year} generic kals'].max()

    print(f" **{dataset} ({year}) Prediction Ranges:**")
    print(f"   - Spatial Kalman Filter: Min = {spatial_min:.2f}, Max = {spatial_max:.2f}")
    print(f"   - Generic Kalman Filter: Min = {generic_min:.2f}, Max = {generic_max:.2f}")
    print("-" * 60)  # Separator for readability

def main():
    for dataset in FACTOR_LIST:
        for year in range(2015, 2021):
            output_histo_path, data_path, spatial_kalman_path, generic_kalman_path  = construct_paths(dataset, year)
            data_df, spatial_kals_df, generic_kals_df = load_data(data_path, DATA_NAMES, spatial_kalman_path, SPATIAL_KALMAN_NAMES, generic_kalman_path, GENERIC_KALMAN_NAMES)
            err_df = calculate_error(data_df, spatial_kals_df, generic_kals_df, year)
            construct_histogram(err_df, output_histo_path, dataset, year)
            # print_prediction_ranges(spatial_kals_df, generic_kals_df, dataset, year)

if __name__ == "__main__":
    main()