import numpy as np
import pandas as pd

# Constants
FACTOR_LIST = ['OD', 'DR', 'SVI Disability']
NUM_COUNTIES = 3143
NUM_YEARS = 7
DATA_COLUMN_NAMES = ['FIPS'] + [f'{yr} data' for yr in range(2014, 2021)]

def construct_path_files(dataset):
    data_file_path = f'Clean Data/{dataset} rates.csv' 
    return data_file_path

def load_data(data_path, data_names):
    data_df = pd.read_csv(data_path, names=data_names, header=0)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].clip(lower=0)  # Ensuring no negative rates
    return data_df

def summarize_data(data_df, dataset_name):
    print(f"\nSummary Statistics for {dataset_name}:\n")
    summary_stats = pd.DataFrame()

    for year in range(2014, 2021):
        year_col = f'{year} data'
        stats = data_df[year_col].describe(percentiles=[0.25, 0.5, 0.75])  # Includes min, 25%, median, 75%, max, mean, std
        stats_df = pd.DataFrame(stats).transpose()
        stats_df['Year'] = year
        summary_stats = pd.concat([summary_stats, stats_df], axis=0)

    # Reordering columns and dropping 'count'
    summary_stats = summary_stats[['Year', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    summary_stats.rename(columns={'50%': 'median'}, inplace=True)

    # Rounding all values to two decimal places
    summary_stats = summary_stats.round(2)

    print(summary_stats.to_string(index=False))

def main():
    for dataset in FACTOR_LIST:
        data_file_path = construct_path_files(dataset)
        data_df = load_data(data_file_path, DATA_COLUMN_NAMES)
        summarize_data(data_df, dataset)

if __name__ == "__main__":
    main()