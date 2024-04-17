import geopandas as gpd
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Constants
FACTOR_LIST = ['OD', 'DR', 'SVI Disability']
SHAPE_PATH = '2020 USA County Shapefile/FIPS_usa.shp'
DATA_NAMES = ['FIPS'] + [f'{yr} Data' for yr in range(2014, 2021)]
KAL_NAMES = ['FIPS'] + [f'{yr} Kal' for yr in range(2014, 2021)]

def construct_file_paths(dataset, year):
    output_map_path = f'Images/Hotspot Accuracy Maps/{dataset}/{year} {dataset} Hotspot Accuracy Map'
    data_path = f'Clean Data/{dataset} rates.csv' 
    kal_path = f'Kalman Predictions/{dataset} Kalman Preds.csv'
    return output_map_path, data_path, kal_path

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    return shape

def load_dataframes(data_path, data_names, kal_path, kal_names):
    data_df = pd.read_csv(data_path, header=0, names=data_names)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].astype(float).clip(lower=0)

    kal_df = pd.read_csv(kal_path, header=0, names=kal_names)
    kal_df['FIPS'] = kal_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    kal_df[kal_names[1:]] = kal_df[kal_names[1:]].astype(float)
    return data_df, kal_df

def compute_density_values(data_df, kal_df, year):
    densities_df = data_df[['FIPS']].copy()

    data_values = data_df[f'{year} Data']
    data_mu, data_sigma = stats.norm.fit(data_values)
    data_densities = []
    for point in data_values:
        density = stats.norm.cdf(point, loc=data_mu, scale=data_sigma)
        data_densities.append(density)
    densities_df[f'{year} Data Density Values'] = data_densities

    kal_values = kal_df[f'{year} Kal']
    kal_mu, kal_sigma = stats.norm.fit(kal_values)
    kal_densities = []
    for point in kal_values:
        density = stats.norm.cdf(point, loc=kal_mu, scale=kal_sigma)
        kal_densities.append(density)
    densities_df[f'{year} Kal Density Values'] = kal_densities
    return densities_df

def merge_data_shape(shape, densities_df):
    return shape.merge(densities_df, on='FIPS')

def plot_heat_map(dataset, shape, year, output_map_path):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    acc = get_accuracy(shape, year)
    title = f'Hotspot Accuracy Map for {dataset} Kalmans in {year}: {acc}%'
    plt.title(title, size=13, weight='bold')

    # Construct the map
    construct_map(shape, fig, main_ax, year)

    # Display and save the map
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

    # Define the insets for coloring
    shapes = [
        (shape[(shape['STATEFP'] != '02') & (shape['STATEFP'] != '15')], main_ax, 'continental'),
        (shape[shape['STATEFP'] == '02'], alaska_ax, 'alaska'),
        (shape[shape['STATEFP'] == '15'], hawaii_ax, 'hawaii') ]
    
    for inset, ax, _ in shapes:
        for _, row in inset.iterrows():
            county = row['FIPS']
            data_density = row[f'{year} Data Density Values']
            kal_density = row[f'{year} Kal Density Values']
            if data_density > .95 and kal_density > .95:
                color = 'orange'
            elif data_density > .95 and kal_density <= .95:
                color = 'black'
            else:
                color = 'lightgrey'
            inset[inset['FIPS'] == county].plot(ax=ax, color=color)

    # Adjust the viewing
    set_view_window(main_ax,alaska_ax,hawaii_ax)

    add_legend(main_ax)

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
    orange_patch = mpatches.Patch(color='darkorange', label='Hotspot hit')
    black_patch = mpatches.Patch(color='black', label='Hotspot miss')
    main_ax.legend(handles=[orange_patch, black_patch], loc='lower right', bbox_to_anchor=(1.00, 0))

def get_accuracy(shape, year):
    data_hot = (shape[f'{year} Data Density Values'] > .95)
    num_data_hot = np.sum(data_hot)
    hot_matches = (shape[f'{year} Data Density Values'] > .95) & (shape[f'{year} Kal Density Values'] > .95)
    num_hot_matches = np.sum(hot_matches)
    acc = round( ((num_hot_matches / num_data_hot) * 100), 2)
    return acc

def main():
    for dataset in FACTOR_LIST:
        for year in range(2014, 2021):
            output_map_path, data_path, kal_path = construct_file_paths(dataset, year)
            shape = load_shapefile(SHAPE_PATH)
            data_df, kal_df = load_dataframes(data_path, DATA_NAMES, kal_path, KAL_NAMES)
            densities_df = compute_density_values(data_df, kal_df, year)
            shape = merge_data_shape(shape, densities_df)
            plot_heat_map(dataset, shape, year, output_map_path)
            print(f'Plot printed for {dataset} in {year}.')

if __name__ == "__main__":
    main()