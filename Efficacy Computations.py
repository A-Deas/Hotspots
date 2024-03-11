from scipy import stats
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
SHAPE_PATH = '/Users/deas/Documents/Research/2020 USA County Shapefile/FIPS_usa.shp'
DATA_NAMES = ['FIPS'] + [f'{yr} Data' for yr in range(2014, 2021)]
KALMAN_NAMES = ['FIPS'] + [f'{yr} Kals' for yr in range(2014, 2021)]

def construct_output_paths(dataset):
    data_path = f'Clean Data/{dataset} rates.csv' 
    kalman_path = f'Kalman Predictions/{dataset} Kalman preds.csv' 
    return data_path, kalman_path

def load_data(data_path, data_names, kalman_path, kalman_names):
    data_df = pd.read_csv(data_path, header=0, names=data_names)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].astype(float).clip(lower=0)

    kals_df = pd.read_csv(kalman_path, header=0, names=kalman_names)
    kals_df['FIPS'] = kals_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    kals_df[kalman_names[1:]] = kals_df[kalman_names[1:]].astype(float)
    return data_df, kals_df

def calculate_err_acc(dataset, data_df, kals_df):
    acc_df = data_df[['FIPS']].copy()

    for year in range(2014, 2021):
        acc_df[f'{year} Absolute Errors'] = abs(kals_df[f'{year} Kals'] - data_df[f'{year} Data'])
        avg_err = np.sum(acc_df[f'{year} Absolute Errors']) / 3143
        max_err = acc_df[f'{year} Absolute Errors'].max()
        print(f'The average error for {dataset} in {year} is {avg_err:.2f}.')
        print(f'The maximum error for {dataset} in {year} is {max_err:.2f}.')

        # Adjusting accuracy calculation
        if max_err == 0:  # Perfect match scenario
            acc_df[f'{year} Accuracy'] = 0.9999
        else:
            acc_df[f'{year} Accuracy'] = 1 - (acc_df[f'{year} Absolute Errors'] / max_err)
            acc_df[f'{year} Accuracy'] = acc_df[f'{year} Accuracy'].apply(lambda x: 0.9999 if x == 1 else (0.0001 if x == 0 else x))
        
        avg_acc = np.sum(acc_df[f'{year} Accuracy']) / 3143
        print(f'The average accuracy for {dataset} in {year} is {avg_acc * 100:.2f}%.\n')

def calculate_hotspot_acc(dataset, data_df, kals_df):
    densities_df = data_df[['FIPS']].copy()

    for year in range(2014, 2021):
        data_values = data_df[f'{year} Data']
        data_mu, data_sigma = stats.norm.fit(data_values)
        data_densities = []
        for point in data_values:
            density = stats.norm.cdf(point, loc=data_mu, scale=data_sigma)
            data_densities.append(density)
        densities_df[f'{year} Data Density Values'] = data_densities

        kal_values = kals_df[f'{year} Kals']
        kal_mu, kal_sigma = stats.norm.fit(kal_values)
        kal_densities = []
        for point in kal_values:
            density = stats.norm.cdf(point, loc=kal_mu, scale=kal_sigma)
            kal_densities.append(density)
        densities_df[f'{year} Kal Density Values'] = kal_densities

        data_hot = (densities_df[f'{year} Data Density Values'] > .95)
        num_data_hot = np.sum(data_hot)
        hot_matches = (densities_df[f'{year} Data Density Values'] > .95) & (densities_df[f'{year} Kal Density Values'] > .95)
        num_hot_matches = np.sum(hot_matches)
        hotspot_acc = round( ((num_hot_matches / num_data_hot) * 100), 2)
        if year == 2020:
            print(f'The hotspot accuracy for {dataset} in {year} is {hotspot_acc}%.\n')
        else:
            print(f'The hotspot accuracy for {dataset} in {year} is {hotspot_acc}%.')

def main():
    for dataset in ['OD','DR','SVI Disability']:
        data_path, kalman_path = construct_output_paths(dataset)
        data_df, kals_df = load_data(data_path, DATA_NAMES, kalman_path, KALMAN_NAMES)
        #calculate_err_acc(dataset, data_df, kals_df)
        calculate_hotspot_acc(dataset, data_df, kals_df)

if __name__ == "__main__":
    main()