import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
FACTOR_LIST = ['OD', 'DR', 'SVI Disability']
SHAPE_PATH = '2020 USA County Shapefile/FIPS_usa.shp'
DATA_NAMES = ['FIPS'] + [f'{yr} data' for yr in range(2014, 2021)]
KALMAN_NAMES = ['FIPS'] + [f'{yr} kals' for yr in range(2014, 2021)]

def construct_output_map_path(dataset, year):
    output_map_path = f'Images/Accuracy Maps/{dataset}/{year} {dataset} Accuracy Map'
    data_path = f'Clean Data/{dataset} rates.csv' 
    kal_path = f'Kalman Predictions/{dataset} Kalman preds.csv' 
    return output_map_path, data_path, kal_path

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    return shape

def load_data(data_path, data_names, kalman_path, kalman_names):
    data_df = pd.read_csv(data_path, header=0, names=data_names)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].astype(float).clip(lower=0)

    kals_df = pd.read_csv(kalman_path, header=0, names=kalman_names)
    kals_df['FIPS'] = kals_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    kals_df[kalman_names[1:]] = kals_df[kalman_names[1:]].astype(float)
    return data_df, kals_df

def calculate_accuracy(data_df, kals_df, year):
    acc_df = data_df[['FIPS']].copy()
    acc_df[f'{year} Absolute Errors'] = abs(kals_df[f'{year} kals'] - data_df[f'{year} data'])
    max_abs_err = acc_df[f'{year} Absolute Errors'].max()

   # Accuracy calculation
    if max_abs_err == 0: # Perfect match
        # Assign slightly less than 1 to remain in cmap interval
        acc_df[f'{year} Accuracy'] = 0.9999 
    else:
        # Calculate accuracy as normal
        acc_df[f'{year} Accuracy'] = 1 - (acc_df[f'{year} Absolute Errors'] / max_abs_err)

        # Then adjust accuracy to 0.9999 if it's exactly 1, and to 0.0001 if it's exactly 0, to remain in cmap interval
        acc_df[f'{year} Accuracy'] = acc_df[f'{year} Accuracy'].apply(lambda x: 0.9999 if x == 1 else (0.0001 if x == 0 else x))
    return acc_df

def merge_data_shape(shape, acc_df):
    return shape.merge(acc_df, on='FIPS')

def plot_accuracy_map(dataset, shape, year, output_map_path):
    """ Plot and save the accuracy map """

    fig, main_ax = plt.subplots(figsize=(10, 5))
    title = f'Accuracy Map for the {dataset} Kalmans in {year}'
    plt.title(title, size=16, weight='bold')

    # Construct the map
    construct_map(shape, fig, main_ax, year)

    plt.savefig(output_map_path, bbox_inches=None, pad_inches=0, dpi=300)
    #plt.show()
    plt.close(fig)

def construct_map(shape, fig, main_ax, year):
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

    # Cmap for accuracy
    num_intervals = 20
    cmap = plt.get_cmap('RdYlGn', num_intervals)

    # Define the insets for coloring
    shapes = [
        (shape[(shape['STATEFP'] != '02') & (shape['STATEFP'] != '15')], main_ax, 'continental'),
        (shape[shape['STATEFP'] == '02'], alaska_ax, 'alaska'),
        (shape[shape['STATEFP'] == '15'], hawaii_ax, 'hawaii') ]

    # Color the maps
    for inset, ax, _ in shapes:
        for _, row in inset.iterrows():
            county = row['FIPS']
            acc = row[f'{year} Accuracy']
            color = cmap(acc)
            inset[inset['FIPS'] == county].plot(ax=ax, color=color)

    # Adjust the viewing
    set_view_window(main_ax,alaska_ax,hawaii_ax)

    # Add the colorbar
    add_colorbar(main_ax, cmap)

def set_view_window(main_ax,alaska_ax,hawaii_ax):
    main_ax.get_xaxis().set_visible(False)
    main_ax.get_yaxis().set_visible(False)
    alaska_ax.set_axis_off()
    hawaii_ax.set_axis_off()
    main_ax.axis('off')

    # Fix window
    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

def add_colorbar(main_ax, cmap):
    """ Accuracy levels """

    color_bounds = np.linspace(0, 100, 21)  # 5% intervals
    norm = BoundaryNorm(color_bounds, cmap.N)
    cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks(color_bounds)
    cbar.set_ticklabels([f'{i}%' for i in color_bounds])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Accuracy Levels', fontsize=10, weight='bold')

def main():
    for dataset in FACTOR_LIST:
        for year in range(2014, 2021):
            output_map_path, data_path, kal_path = construct_output_map_path(dataset, year)
            shape = load_shapefile(SHAPE_PATH)
            data_df, kals_df = load_data(data_path, DATA_NAMES, kal_path, KALMAN_NAMES)
            acc_df = calculate_accuracy(data_df, kals_df, year)
            shape = merge_data_shape(shape, acc_df)
            plot_accuracy_map(dataset, shape, year, output_map_path)
            print(f'Plot printed for {dataset} in {year}.')

if __name__ == "__main__":
    main()