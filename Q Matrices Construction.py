import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.wkt import loads
from math import radians, sin, cos, sqrt, atan2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
SHAPE_PATH = '2020 USA County Shapefile/FIPS_usa.shp'
CRS_TARGET = 'EPSG:4326' # CRS for longitude and latitude coordinates
NUM_COUNTIES = 3143  # In the United States as of 2020
MAX_DISTANCE = 4572  # Precomputed maximum distance between population centroids in continental U.S. measured in kilometers
OD_DECAY_RATE = 0.00125 # for 50% correlation at a threshold of 555 kilometers distance from Polk TN to Franklin OH
DR_DECAY_RATE = 0.00375 # for 50% correlation at a threshold of 185 (555/3) kilometers
SVI_DECAY_RATE = 0.000937 # for 50% correlation at a threshold of 740 kilometers approx distance from Hildalgo NM to Union NM

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371.0 * c  # Radius of earth in kilometers 
    return distance

def load_shapefile(path):
    """ Load and prepare the shapefile for processing """

    shape = gpd.read_file(path)
    shape = shape.to_crs(CRS_TARGET)
    shape['Pop Cents'] = shape['Pop Cents'].apply(lambda x: loads(x)) # Bring the Long/Lat coordinates back to points (saved as strings in shape)
    return shape

def compute_Q_od(shape, num_counties, decay_rate):
    Q_od = np.zeros((num_counties, num_counties))  # Initialize the Q matrix

    for i in range(num_counties):
        for j in range(i+1, num_counties):
            lat1, lon1 = shape.iloc[i]['Pop Cents'].y, shape.iloc[i]['Pop Cents'].x
            lat2, lon2 = shape.iloc[j]['Pop Cents'].y, shape.iloc[j]['Pop Cents'].x
            distance = haversine(lat1, lon1, lat2, lon2)
            Q_od[i, j] = np.exp(-decay_rate * distance)
            Q_od[j, i] = Q_od[i, j]  # Exploit symmetry

    np.fill_diagonal(Q_od, 1)
    return Q_od

def compute_Q_dr(shape, num_counties, decay_rate):
    Q_dr = np.zeros((num_counties, num_counties))  # Initialize the Q matrix

    for i in range(num_counties):
        for j in range(i+1, num_counties):
            lat1, lon1 = shape.iloc[i]['Pop Cents'].y, shape.iloc[i]['Pop Cents'].x
            lat2, lon2 = shape.iloc[j]['Pop Cents'].y, shape.iloc[j]['Pop Cents'].x
            distance = haversine(lat1, lon1, lat2, lon2)
            Q_dr[i, j] = np.exp(-decay_rate * distance)
            Q_dr[j, i] = Q_dr[i, j]  # Exploit symmetry

    # Fill the diagonal with 1s. This step can be optional based on how you initialize your matrix and your specific needs.
    np.fill_diagonal(Q_dr, 1)
    return Q_dr

def compute_Q_svi(shape, num_counties, decay_rate):
    Q_svi = np.zeros((num_counties, num_counties))  # Initialize the Q matrix

    for i in range(num_counties):
        for j in range(i+1, num_counties):
            lat1, lon1 = shape.iloc[i]['Pop Cents'].y, shape.iloc[i]['Pop Cents'].x
            lat2, lon2 = shape.iloc[j]['Pop Cents'].y, shape.iloc[j]['Pop Cents'].x
            distance = haversine(lat1, lon1, lat2, lon2)
            Q_svi[i, j] = np.exp(-decay_rate * distance)
            Q_svi[j, i] = Q_svi[i, j]  # Exploit symmetry

    # Fill the diagonal with 1s. This step can be optional based on how you initialize your matrix and your specific needs.
    np.fill_diagonal(Q_svi, 1)
    return Q_svi

def save_matrix(matrix, filename):
    pd.DataFrame(matrix).to_csv(filename, index=False, header=False)

def main():
    shape = load_shapefile(SHAPE_PATH)
    
    # Compute and save Q_OD matrix
    Q_od = compute_Q_od(shape, NUM_COUNTIES, OD_DECAY_RATE)
    save_matrix(Q_od, 'Covariance Matrices/Q_OD.csv')

    # Compute and save Q_DR matrix
    Q_dr = compute_Q_dr(shape, NUM_COUNTIES, DR_DECAY_RATE)
    save_matrix(Q_dr, 'Covariance Matrices/Q_DR.csv')

    # Compute and save Q_SVI matrix
    Q_svi = compute_Q_svi(shape, NUM_COUNTIES, SVI_DECAY_RATE)
    save_matrix(Q_svi, 'Covariance Matrices/Q_SVI.csv')

if __name__ == "__main__":
    main()
