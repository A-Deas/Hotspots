import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Constants
FACTOR_LIST = ['OD', 'DR', 'SVI Disability']
DATA_NAMES = ['FIPS'] + [f'{yr} data' for yr in range(2014, 2021)]
KALMAN_NAMES = ['FIPS'] + [f'{yr} kals' for yr in range(2014, 2021)]

def construct_output_histo_path(dataset, year):
    output_histo_path = f'Images/Error Histograms/{dataset}/{year} {dataset} Error Histogram'
    data_path = f'Clean Data/{dataset} rates.csv' 
    kalman_path = f'Kalman Predictions/{dataset} Kalman preds.csv' 
    return output_histo_path, data_path, kalman_path

def load_data(data_path, data_names, kalman_path, kalman_names):
    data_df = pd.read_csv(data_path, header=0, names=data_names)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].astype(float).clip(lower=0)

    kals_df = pd.read_csv(kalman_path, header=0, names=kalman_names)
    kals_df['FIPS'] = kals_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    kals_df[kalman_names[1:]] = kals_df[kalman_names[1:]].astype(float)
    return data_df, kals_df

def calculate_error(data_df, kals_df, year):
    err_df = data_df[['FIPS']].copy()
    err_df[f'{year} Absolute Errors'] = abs(kals_df[f'{year} kals'] - data_df[f'{year} data'])
    return err_df

def print_top_errors(err_df, year):
    top_errors_df = err_df.sort_values(by=f'{year} Absolute Errors', ascending=False).head(5)
    top_errors_df_reset = top_errors_df.reset_index(drop=True)
    print(top_errors_df_reset)


def construct_histogram(dataset, err_df, output_histo_path, year):
    plt.figure(figsize=(8, 6))
    errors = err_df[f'{year} Absolute Errors']
    max_error = errors.max().round(2)
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel('Absolute Error', fontsize=12, weight='bold')
    plt.ylabel('Frequency', fontsize=12, weight='bold')

    if dataset == 'OD':
        size = 15
        if year == 2014:
            tick_positions = np.arange(0, 1, 1)
        elif 2014 < year < 2020:
            tick_positions = np.arange(0, max_error+1, 1)
        else: 
            tick_positions = np.arange(0, max_error+5, 5)
    elif dataset == 'DR':
        size = 15
        if year == 2014:
            tick_positions = np.arange(0, 1, 1)
        elif 2014 < year < 2018:
            tick_positions = np.arange(0, max_error+1, 1)
        elif 2018 <= year <= 2019:
            tick_positions = np.arange(0, max_error+5, 5)
        else: 
            tick_positions = np.arange(0, max_error+30, 30)
    elif dataset.startswith('SVI'):
        size = 13
        if year == 2014:
            tick_positions = np.arange(0, 1, 1)
        elif 2014 < year < 2020:
            tick_positions = np.arange(0, max_error+1, 1)
        else: 
            tick_positions = np.arange(0, max_error+5, 5)

    tick_labels = [str(int(x)) for x in tick_positions]
    plt.xticks(tick_positions, tick_labels) 

    title = f'Absolute Errors for the {dataset} Kalmans in {year}'
    plt.title(title, size=size, weight='bold')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(output_histo_path, bbox_inches=None, pad_inches=0, dpi=300)
    #plt.show()
    plt.close()

def main():
    for dataset in FACTOR_LIST:
        for year in range(2014, 2021):
            output_histo_path, data_path, kalman_path = construct_output_histo_path(dataset, year)
            data_df, kals_df = load_data(data_path, DATA_NAMES, kalman_path, KALMAN_NAMES)
            err_df = calculate_error(data_df, kals_df, year)
            #print_top_errors(err_df, year)
            construct_histogram(dataset, err_df, output_histo_path, year)
            print(f'Histogram printed for {dataset} in {year}.')

if __name__ == "__main__":
    main()
