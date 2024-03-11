import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from scipy.stats import norm
import warnings 
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
SHAPE_PATH = '/Users/deas/Documents/Research/2020 USA County Shapefile/FIPS_usa.shp'
DATA_NAMES = ['FIPS'] + [f'{yr} data' for yr in range(2014, 2021)]
KALMAN_NAMES = ['FIPS'] + [f'{yr} kals' for yr in range(2014, 2021)]

def construct_paths(dataset, year):
    output_ci_path = f'/Users/deas/Documents/Research/Paper 1/CI Maps/{dataset}/{year} {dataset} 95% CI Map'
    data_path = f'Clean Data/{dataset} rates.csv' 
    kalman_path = f'Kalman Predictions/{dataset} Kalman preds.csv' 
    return output_ci_path, data_path, kalman_path

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    return shape

def load_data(dataset, data_path, data_names, kalman_path, kalman_names):
    data_df = pd.read_csv(data_path, header=0, names=data_names)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].astype(float).clip(lower=0)

    kals_df = pd.read_csv(kalman_path, header=0, names=kalman_names)
    kals_df['FIPS'] = kals_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    kals_df[kalman_names[1:]] = kals_df[kalman_names[1:]].astype(float)
    return data_df, kals_df

def calculate_confidence_metrics(dataset, data_df, year):
    fitted_normal = norm.fit(data_df[f'{year} data'])
    mean = fitted_normal[0]
    std = fitted_normal[1]
    print(f'mean = {mean}')
    print(f'std = {std}')
    return mean, std

def merge_dataframes(shape, kals_df):
    shape = shape.merge(kals_df, on='FIPS')
    return shape

def plot_accuracy_map(dataset, shape, year, mean, std, output_ci_path):
    """ Plot and save the accuracy map """

    fig, main_ax = plt.subplots(figsize=(10, 5))
    title = f'95% CI Map for the {dataset} Kalmans in {year}'
    plt.title(title, size=14, weight='bold')

    # Construct the map
    construct_map(shape, fig, main_ax, year, mean, std)

    # Display and save the map
    plt.savefig(output_ci_path, bbox_inches=None, pad_inches=0, dpi=300)
    plt.show()
    plt.close(fig)

def construct_map(shape, fig, main_ax, year, mean, std):
    # Alaska and Hawaii insets
    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4]) 
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])  
    
    # Plot state boundaries
    state_boundaries = shape.dissolve(by='STATEFP', as_index=False)
    state_boundaries.boundary.plot(ax=main_ax, edgecolor='black', linewidth=.5)

    alaska_state = state_boundaries[state_boundaries['STATEFP'] == '02']
    alaska_state.boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=.5)

    hawaii_state = state_boundaries[state_boundaries['STATEFP'] == '15']
    hawaii_state.boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=.5)

    # Define the insets for coloring
    shapes = [
        (shape[(shape['STATEFP'] != '02') & (shape['STATEFP'] != '15')], main_ax, 'continental'),
        (shape[shape['STATEFP'] == '02'], alaska_ax, 'alaska'),
        (shape[shape['STATEFP'] == '15'], hawaii_ax, 'hawaii') ]
    
    # Color the maps
    z = 1.960 # 95% CI
    count = 0
    for inset, ax, _ in shapes:
        for _, row in inset.iterrows():
            county = row['FIPS']
            kal_value = row[f'{year} kals']

            lower_bound = mean - z * std
            upper_bound = mean + z * std

            if lower_bound <= kal_value <= upper_bound:
                color = 'wheat'
            elif kal_value < lower_bound:
                color = 'blue'
                count += 1
            elif kal_value > upper_bound:
                color = 'red'
                count += 1
            inset[inset['FIPS'] == county].plot(ax=ax, color=color)

    # Adjust the viewing
    set_view_window(main_ax,alaska_ax,hawaii_ax)

    # Add the colorbar
    add_legend(main_ax)
    out = (count / 3143) * 100
    out = round(out, 2)
    print(f'Percentage outside of CI is: {out}')

def set_view_window(main_ax,alaska_ax,hawaii_ax):
    main_ax.get_xaxis().set_visible(False)
    main_ax.get_yaxis().set_visible(False)
    alaska_ax.set_axis_off()
    hawaii_ax.set_axis_off()
    main_ax.axis('off')

    # Fix window
    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

def add_legend(main_ax):
    orange_patch = mpatches.Patch(color='wheat', label='Within 95% C.I.')
    purple_patch = mpatches.Patch(color='purple', label='Outside 95% C.I.')
    main_ax.legend(handles=[orange_patch, purple_patch], loc='lower right', bbox_to_anchor=(1.05, 0))

def main():
    for dataset in ['OD','DR','SVI Disability']:
        for year in range(2020, 2021):
            output_ci_path, data_path, kalman_path = construct_paths(dataset, year)
            shape = load_shapefile(SHAPE_PATH)
            data_df, kals_df = load_data(dataset, data_path, DATA_NAMES, kalman_path, KALMAN_NAMES)
            mean, std = calculate_confidence_metrics(dataset, data_df, year)
            shape = merge_dataframes(shape, kals_df)
            plot_accuracy_map(dataset, shape, year, mean, std, output_ci_path)

if __name__ == "__main__":
    main()