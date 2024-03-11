import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Constants
SHAPEFILE_PATH = '/Users/deas/Documents/Research/2020 USA County Shapefile/cb_2020_us_county_20m.shp'
POP_CENTROIDS_CSV = 'Dirty Data/Dirty Population Centroids.csv'
EXCLUDE_TERITORIES = ['03', '07', '14', '43', '52', '72']
FILTERED_SHAPEFILE_PATH = '/Users/deas/Documents/Research/2020 USA County Shapefile/FIPS_usa.shp'

def load_data(shapefile_path, pop_centroids_csv, exclude_territories):
    """ Clean the files by excluding the territories we don't want """
    shape = gpd.read_file(shapefile_path, dtype={'STATEFP': str, 'COUNTYFP': str})
    pop_df = pd.read_csv(pop_centroids_csv, dtype={'STATEFP': str, 'COUNTYFP': str})
    shape = shape[~shape['STATEFP'].isin(exclude_territories)]
    pop_df = pop_df[~pop_df['STATEFP'].isin(exclude_territories)]
    return shape, pop_df

def create_fips_codes(shape, pop_df):
    """ Construct the 5 digit FIPS codes """
    for df in (shape, pop_df):
        df['FIPS'] = df['STATEFP'] + df['COUNTYFP']
        df.sort_values('FIPS', inplace=True)
    return shape, pop_df

def check_fips_codes(shape, pop_df):
    """ Check tha FIPS codes are a match """
    shape_fips_in_pop = shape['FIPS'].isin(pop_df['FIPS']).all()
    pop_fips_in_shape = pop_df['FIPS'].isin(shape['FIPS']).all()
    match = shape_fips_in_pop == pop_fips_in_shape
    print("FIPS codes are a match? ", match)

def merge_population_centroids(shape, pop_df):
    """ Merge the population centroids into the shapefile """
    pop_df['Pop Cents'] = pop_df.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)
    shape = shape.merge(pop_df[['FIPS', 'Pop Cents']], on='FIPS', how='left')
    shape['Pop Cents'] = shape['Pop Cents'].astype(str) # Shapefiles can't have more than one geometry
    return shape

def save_filtered_shapefile(shape, filtered_shapefile_path):
    shape.to_file(filtered_shapefile_path)

def main():
    shape, pop_df = load_data(SHAPEFILE_PATH, POP_CENTROIDS_CSV, EXCLUDE_TERITORIES)
    shape, pop_df = create_fips_codes(shape, pop_df)
    check_fips_codes(shape, pop_df)
    shape = merge_population_centroids(shape, pop_df)
    save_filtered_shapefile(shape, FILTERED_SHAPEFILE_PATH)

if __name__ == "__main__":
    main()
