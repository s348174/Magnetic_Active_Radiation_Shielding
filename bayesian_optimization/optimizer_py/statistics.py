import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def read_energy_data(file_path):
    """
    Reads energy data from a CSV file and returns a DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file containing energy data.
    
    Returns:
    np.ndarray: A DataFrame containing the energy steps (column 1).
    pd.DataFrame: A DataFrame containing the energy data (columns 1 to end).
    """
    full_dataset = pd.read_csv(file_path)
    energy_steps = np.array(full_dataset.iloc[:, 0].values)
    data = full_dataset.iloc[:, 1:]
    return energy_steps, data

def build_log_scaled_dataset(energy_steps, data):
    """
    Builds a log-scaled dataset from the energy steps and data.
    
    Parameters:
    energy_steps (np.ndarray): A numpy array containing the energy steps.
    data (pd.DataFrame): A DataFrame containing the energy data.
    
    Returns:
    pd.DataFrame: A new DataFrame with log-scaled energy steps and data.
    """
    log_energy_steps = np.log(energy_steps)
    normalized_data = (data - data.mean()) / data.std()
    log_scaled_dataset = pd.DataFrame(normalized_data, columns=data.columns)
    log_scaled_dataset.insert(0, 'log_energy_steps', log_energy_steps)
    return log_scaled_dataset

def compute_column_means(log_scaled_dataset):
    """
    Computes the mean of each column in the DataFrame.
    
    Parameters:
    log_scaled_dataset (pd.DataFrame): The input DataFrame containing log-scaled energy data.
    
    Returns:
    A numpy vector containing the mean of each column.
    """
    means = []
    for column in log_scaled_dataset.columns[1:]:  # Skip the first column (log_energy_steps)
        mean = 0
        for i in range(log_scaled_dataset[column].shape[0]):
            mean += log_scaled_dataset[column][i]*log_scaled_dataset['log_energy_steps'][i]
        means.append(mean)
    return np.array(means)

def compute_column_variances(log_scaled_dataset):
    """
    Computes the variance of each column in the DataFrame.
    
    Parameters:
    log_scaled_dataset (pd.DataFrame): The input DataFrame containing log-scaled energy data.
    
    Returns:
    A numpy vector containing the variance of each column.
    """
    variances = []
    means = compute_column_means(log_scaled_dataset)
    for column in log_scaled_dataset.columns[1:]:  # Skip the first column (log_energy_steps)
        mean = means[log_scaled_dataset.columns.get_loc(column) - 1]  # Get the mean for the current column
        mean_square = 0
        for i in range(log_scaled_dataset[column].shape[0]):
            mean_square = log_scaled_dataset['log_energy_steps'].iloc[i] ** 2 * log_scaled_dataset[column].iloc[i]
        variances.append(mean_square - mean ** 2)
    return np.array(variances)

def test_normality(log_scaled_data):
    """
    Fits a normal distribution to each column of the log-scaled dataset using empirical params."""
    means = compute_column_means(log_scaled_data)
    variances = compute_column_variances(log_scaled_data)
    stds = np.sqrt(variances)

    for column in log_scaled_data.columns[1:]:  # Skip the first column (log_energy_steps)
        # Test if the points lie on a normal ditribution
        cumdiff = 0
        reldiff = 0
        mean = means[log_scaled_data.columns.get_loc(column) - 1]  # Get the mean for the current column
        std = stds[log_scaled_data.columns.get_loc(column) - 1] 
         # Loop through each data point and compute the predicted value from the normal distribution
        for i in range(log_scaled_data['log_energy_steps'].shape[0]):
            predicted = mean + std * stats.norm.pdf(log_scaled_data['log_energy_steps'][i])
            actual = log_scaled_data[column][i]
            cumdiff += abs(predicted - actual)
            reldiff += abs(predicted - actual) / (abs(actual) + 1e-8)  # Avoid division by zero
        mean_error = cumdiff / log_scaled_data['log_energy_steps'].shape[0]
        relative_error = reldiff / log_scaled_data['log_energy_steps'].shape[0]
        print(f"Column: {column}, Cumdiff: {cumdiff:.4f}, Mean Error: {mean_error:.4f}, Relative Error: {relative_error:.4f}")

def test_stats(filepath='../data/flux_data_clean.csv'):
    energy_steps, data = read_energy_data(filepath)
    print("Energy data read successfully.")
    log_scaled_data = build_log_scaled_dataset(energy_steps, data)
    print("Means and variances of log-scaled dataset:")
    means = compute_column_means(log_scaled_data)
    variances = compute_column_variances(log_scaled_data)
    for i, column in enumerate(log_scaled_data.columns[1:]):  # Skip the first column (log_energy_steps)
        print(f"Column: {column}, Mean: {means[i]:.4f}, Variance: {variances[i]:.4f}")
    print("Log-scaled dataset built successfully.")
    print("First few rows of the log-scaled dataset:")
    print(log_scaled_data.head())
    test_normality(log_scaled_data)

if __name__ == "__main__":
    test_stats()