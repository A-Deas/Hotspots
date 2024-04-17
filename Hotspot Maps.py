import geopandas as gpd
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Constants
FACTOR_LIST = ['OD', 'DR', 'SVI Disability']
SHAPE_PATH = '2020 USA County Shapefile/FIPS_usa.shp'
KAL_NAMES = ['FIPS'] + [f'{yr} Kals' for yr in range(2014, 2021)]

def construct_file_paths(dataset, year):
    output_map_path = f'Images/Hotspot Maps/{dataset}/{year} {dataset} Hotspot Map'
    kal_path = f'Kalman Predictions/{dataset} Kalman Preds.csv'
    return output_map_path, kal_path

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    return shape

def load_kals(kal_path, kal_names):
    kals_df = pd.read_csv(kal_path, header=0, names=kal_names)
    kals_df['FIPS'] = kals_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    kals_df[kal_names[1:]] = kals_df[kal_names[1:]].astype(float)
    return kals_df

def compute_density_values(kals_df, year):
    yearly_kals = kals_df[f'{year} Kals']
    mu, sigma = stats.norm.fit(yearly_kals)

    density_values = []
    for point in yearly_kals:
        density = stats.norm.cdf(point, loc=mu, scale=sigma)
        density_values.append(density)

    kals_df[f'{year} Density Values'] = density_values
    return kals_df

def merge_data_shape(shape, kals_df):
    return shape.merge(kals_df, on='FIPS')

def plot_heat_map(dataset, shape, year, output_map_path):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    title = f'Hotspot Map for the {dataset} Kalmans in {year}'
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

    # Color the maps
    percentiles = []
    for i in range(5,101,5):
        perc = i / 100
        percentiles.append(perc)
    
    for inset, ax, _ in shapes:
        for _, row in inset.iterrows():
            county = row['FIPS']
            density_value = row[f'{year} Density Values']
            if density_value > .95:
                color = 'maroon'
            else:
                color = 'lightgrey'
            inset[inset['FIPS'] == county].plot(ax=ax, color=color)

    # Adjust the viewing
    set_view_window(main_ax,alaska_ax,hawaii_ax)

    # Add the legend
    add_legend(main_ax)

def define_colors():
    colors = ['midnightblue','mediumblue','blue','royalblue', 'cornflowerblue',
          'lightblue','powderblue','paleturquoise','lightcyan','azure',
          'papayawhip','bisque','wheat','orange','darkorange',
          'orangered','red','firebrick','darkred','maroon']
    return colors

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
    maroon_patch = mpatches.Patch(color='maroon', label='Hotspot')
    main_ax.legend(handles=[maroon_patch], loc='lower right', bbox_to_anchor=(0.95, 0))

def main():
    for dataset in FACTOR_LIST:
        for year in range(2014, 2021):
            output_map_path, kal_path = construct_file_paths(dataset, year)
            shape = load_shapefile(SHAPE_PATH)
            kals_df = load_kals(kal_path, KAL_NAMES)
            kals_df = compute_density_values(kals_df, year)
            shape = merge_data_shape(shape, kals_df)
            plot_heat_map(dataset, shape, year, output_map_path)
            print(f'Plot printed for {dataset} in {year}.')

if __name__ == "__main__":
    main()