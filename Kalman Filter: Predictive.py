import numpy as np
import pandas as pd

# Constants
FACTOR_LIST = ['OD', 'DR', 'SVI Disability']
NUM_COUNTIES = 3143
NUM_YEARS = 7
DATA_COLUMN_NAMES = ['FIPS'] + [f'{yr} data' for yr in range(2014, 2021)]

def construct_path_files(dataset):
    data_file_path = f'Clean Data/{dataset} rates.csv' 
    q_matrix_path = f'Covariance Matrices/Q_{dataset}.csv' 
    output_file_path = f'Kalman Predictions/{dataset} Kalman Preds.csv'
    kal_names = [f'{year} {dataset} Ests' if year < 2020 else f'{year} {dataset} Preds' for year in range(2014, 2021)]
    return data_file_path, q_matrix_path, output_file_path, kal_names

def load_data(data_path, data_names):
    data_df = pd.read_csv(data_path, names=data_names, header=0)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].clip(lower=0)
    return data_df

def initialize_matrices(dataset, q_matrix_path, num_counties):
    F = np.eye(num_counties)
    H = np.eye(num_counties)
    R = np.eye(num_counties) * 0.01
    if dataset.startswith('SVI'):
        Q = pd.read_csv('Covariance Matrices/Q_SVI.csv', header=None).to_numpy()
    else:
        Q = pd.read_csv(q_matrix_path, header=None).to_numpy()
    return F, H, R, Q

def run_kalman_filter(num_counties, num_years, data_df, F, H, R, Q):
    updated_rates = np.zeros((num_years, num_counties))
    updated_rates_covariances = np.zeros((num_years, num_counties, num_counties))
    initial_state_guess = data_df['2014 data'].values
    x = initial_state_guess
    updated_rates[0,:] = x
    P = np.eye(num_counties) * 0.01  # Initial state uncertainty

    for t in range(1, num_years-1):
        x, P, y, K = kalman_estimate_update(num_counties, x, P, F, Q, H, R, data_df, t)
        updated_rates[t, :] = x
        updated_rates_covariances[t, :, :] = P

    # Use 2019 data and Kalman gain to make 2020 predictions
    x_pred = x + (K @ y)
    P_pred = (np.eye(num_counties) - K @ H) @ P 
    updated_rates[num_years-1, :] = x_pred
    updated_rates_covariances[num_years-1, :, :] = P_pred

    return updated_rates, updated_rates_covariances

def kalman_estimate_update(num_counties, x, P, F, Q, H, R, data_df, t):
    year = 2014 + t
    x = F @ x  # Predicted state estimate
    P = F @ P @ F.T + Q  # Predicted estimate covariance
    y = data_df[f'{year} data'].values - H @ x  # Pre-fit residual
    S = H @ P @ H.T + R  # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    x += K @ y  # Updated state estimate
    P = (np.eye(num_counties) - K @ H) @ P  # Updated estimate covariance
    return x, P, y, K

def save_results(updated_rates, data_df, column_names, output_path):
    updated_rates_df = pd.DataFrame(updated_rates.T, columns=column_names) 
    updated_rates_df['FIPS'] = data_df['FIPS']
    updated_rates_df = updated_rates_df[['FIPS'] + column_names]
    updated_rates_df.round(2).to_csv(output_path, index=False)

def main():
    for dataset in FACTOR_LIST:
        data_file_path, q_matrix_path, output_file_path, kal_names = construct_path_files(dataset)
        data_df = load_data(data_file_path, DATA_COLUMN_NAMES)
        F, H, R, Q = initialize_matrices(dataset, q_matrix_path, NUM_COUNTIES)
        updated_rates, _ = run_kalman_filter(NUM_COUNTIES, NUM_YEARS, data_df, F, H, R, Q)
        save_results(updated_rates, data_df, kal_names, output_file_path)

if __name__ == "__main__":
    main()
