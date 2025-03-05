import numpy as np
import pandas as pd

# Constants
FACTOR_LIST = ['OD', 'DR', 'SVI Disability']
NUM_COUNTIES = 3143
DATA_COLUMN_NAMES = ['FIPS'] + [f'{yr} data' for yr in range(2014, 2021)]

def construct_output_path(dataset, init_year):
    output_path = f'KF PoLD/PoLD Predictions/{dataset}/{dataset}_initialized_at_{init_year}.csv' 
    data_path = f'Clean Data/{dataset} rates.csv' 
    q_matrix_path = f'Covariance Matrices/Q_{dataset}.csv' 
    return output_path, data_path, q_matrix_path

def load_data(data_path, data_names):
    data_df = pd.read_csv(data_path, names=data_names, header=0)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].clip(lower=0)
    return data_df

def initialize_matrices(dataset, num_counties, q_matrix_path):
    F = np.eye(num_counties)
    H = np.eye(num_counties)
    R = np.eye(num_counties) * 0.01
    if dataset.startswith('SVI'):
        Q = pd.read_csv('Covariance Matrices/Q_SVI.csv', header=None).to_numpy()
    else:
        Q = pd.read_csv(q_matrix_path, header=None).to_numpy()
    return F, H, R, Q

def run_kalman_filter(num_counties, init_year, data_df, F, H, R, Q):
    """ Run Kalman Filter starting from `init_year` and predicting up to 2020. """
    
    num_years = 2020 - init_year + 1  # Number of years from init_year to 2020
    updated_rates = np.zeros((num_years, num_counties))
    updated_rates_covariances = np.zeros((num_years, num_counties, num_counties))

    # Initialize using data from `init_year`
    x = data_df[f'{init_year} data'].values
    updated_rates[0, :] = x
    P = np.eye(num_counties) * 0.01  # Initial state uncertainty

    if init_year < 2019:
        # Apply Kalman filter updates for observed data
        for t in range(1, num_years-1):  # Update until 2019
            x, P, y, K = kalman_estimate_update(num_counties, x, P, F, Q, H, R, data_df, t, init_year)
            updated_rates[t, :] = x
            updated_rates_covariances[t, :, :] = P

        # Use latest Kalman gain to predict 2020
        x = x + (K @ y) + np.random.multivariate_normal(mean=np.zeros(num_counties), cov=Q)
        P = (np.eye(num_counties) - K @ H) @ P 
        updated_rates[num_years-1, :] = x
        updated_rates_covariances[num_years-1, :, :] = P

    elif init_year == 2019:
        # Predict straight from 2019 data
        x = F @ x + np.random.multivariate_normal(mean=np.zeros(num_counties), cov=Q)  # Predicted state
        P = F @ P @ F.T + Q  # Predicted estimate covariance
        y = data_df[f'2019 data'].values - H @ x  # Pre-fit residual
        S = H @ P @ H.T + R  # Residual covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
        x += K @ y  # Updated state estimate
        P = (np.eye(num_counties) - K @ H) @ P  # Updated estimate covariance
        updated_rates[1, :] = x
        updated_rates_covariances[1, :, :] = P

    return updated_rates, updated_rates_covariances

def kalman_estimate_update(num_counties, x, P, F, Q, H, R, data_df, t, init_year):
    year = init_year + t
    """ Kalman filter update step for a specific year """
    x = F @ x + np.random.multivariate_normal(mean=np.zeros(num_counties), cov=Q)  # Predicted state
    P = F @ P @ F.T + Q  # Predicted estimate covariance
    y = data_df[f'{year} data'].values - H @ x  # Pre-fit residual
    S = H @ P @ H.T + R  # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    x += K @ y  # Updated state estimate
    P = (np.eye(num_counties) - K @ H) @ P  # Updated estimate covariance
    return x, P, y, K

def save_results(updated_rates, data_df, output_path, init_year):
    """ Save results with adjusted column names reflecting initialization year. """
    output_columns = [f'{yr} kals' for yr in range(init_year, 2021)]
    
    updated_rates_df = pd.DataFrame(updated_rates.T, columns=output_columns) 
    updated_rates_df['FIPS'] = data_df['FIPS']
    updated_rates_df = updated_rates_df[['FIPS'] + output_columns]
    updated_rates_df.round(2).to_csv(output_path, index=False)

def main():
    np.random.seed(42) # set random seed for reproducibility 

    for dataset in FACTOR_LIST:
        for init_year in range(2014, 2020):  # From 2014 to 2019
            output_path, data_path, q_matrix_path = construct_output_path(dataset, init_year)
            data_df = load_data(data_path, DATA_COLUMN_NAMES)
            F, H, R, Q = initialize_matrices(dataset, NUM_COUNTIES, q_matrix_path)
            updated_rates, _ = run_kalman_filter(NUM_COUNTIES, init_year, data_df, F, H, R, Q)
            save_results(updated_rates, data_df, output_path, init_year)
            print(f'KF trained on {dataset} and initialized at {init_year} complete.')

if __name__ == "__main__":
    main()