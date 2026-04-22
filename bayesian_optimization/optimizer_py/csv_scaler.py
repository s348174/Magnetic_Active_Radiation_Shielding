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
    # Map log_energy_steps to the same interval as the original energy steps for better visualization
    log_energy_steps = (log_energy_steps - log_energy_steps.min()) / (log_energy_steps.max() - log_energy_steps.min()) * (energy_steps.max() - energy_steps.min()) + energy_steps.min()
    log_scaled_data = np.log(data.replace(0, np.nan))  # Replace zeros with NaN to avoid log(0)
    # Map log_scaled_data to the same interval as the original data for better visualization
    log_scaled_data = (log_scaled_data - log_scaled_data.min()) / (log_scaled_data.max() - log_scaled_data.min()) * (data.max().max() - data.min().min()) + data.min().min()
    log_scaled_dataset = pd.DataFrame(log_scaled_data, columns=data.columns)
    log_scaled_dataset.insert(0, 'Energy', log_energy_steps)
    return log_scaled_dataset

def decompress_log_scaled_samples(log_scaled_samples, data):
    """
    Decompresses log-scaled samples back to their original scale.
    
    Parameters:
    log_scaled_samples (dict[str, np.ndarray]): A dictionary containing log-scaled samples for each particle.
    
    Returns:
    dict[str, np.ndarray]: A dictionary containing decompressed samples for each particle.
    """
    decompressed_samples = {}
    for particle, samples in log_scaled_samples.items():
        # Remap log-scaled samples back to the same interval as the log data
        log_min = np.log(data[particle].replace(0, np.nan)).min()
        log_max = np.log(data[particle].replace(0, np.nan)).max()
        samples = (samples - samples.min()) / (samples.max() - samples.min()) * (log_max - log_min) + log_min
        decompressed_samples[particle] = np.exp(samples)
    return decompressed_samples

def build_energy_scaled_dataset(energy_steps, data):
    """
    Builds an energy-scaled dataset from the energy steps and data.
    
    Parameters:
    energy_steps (np.ndarray): A numpy array containing the energy steps.
    data (pd.DataFrame): A DataFrame containing the energy data.
    
    Returns:
    pd.DataFrame: A new DataFrame with energy-scaled data.
    """
    log_energy_steps = np.log(energy_steps)
    # Map log_energy_steps to the same interval as the original energy steps for better visualization
    log_energy_steps = (log_energy_steps - log_energy_steps.min()) / (log_energy_steps.max() - log_energy_steps.min()) * (energy_steps.max() - energy_steps.min()) + energy_steps.min()
    log_scaled_dataset = pd.DataFrame(data, columns=data.columns)
    log_scaled_dataset.insert(0, 'Energy', log_energy_steps)
    return log_scaled_dataset

def plot_log_scaled_data(log_scaled_dataset):
    """
    Plots the log-scaled dataset.
    
    Parameters:
    log_scaled_dataset (pd.DataFrame): The input DataFrame containing log-scaled energy data.
    """
    plt.figure(figsize=(10, 6))
    for column in log_scaled_dataset.columns[1:]:  # Skip the first column (log_energy_steps)
        plt.plot(log_scaled_dataset['Energy'], log_scaled_dataset[column], label=column)
    plt.xlabel('Log Energy Steps')
    plt.ylabel('Normalized Energy Data')
    plt.title('Log-Scaled Energy Data')
    plt.legend()
    plt.grid()
    plt.show()

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
        num = np.trapezoid(log_scaled_dataset['Energy'].values*log_scaled_dataset[column].values, log_scaled_dataset['Energy'].values)
        denom = np.trapezoid(log_scaled_dataset[column].values, log_scaled_dataset['Energy'].values)
        means.append(num / denom if denom != 0 else 0)
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
        num = np.trapezoid((log_scaled_dataset['Energy'].values)**2 * log_scaled_dataset[column].values, log_scaled_dataset['Energy'].values)
        denom = np.trapezoid(log_scaled_dataset[column].values, log_scaled_dataset['Energy'].values)
        mean_sq = num / denom if denom != 0 else 0
        variances.append(mean_sq - mean**2)
    return np.array(variances)
    
if __name__ == "__main__":
    filepath = '../data/flux_data_clean.csv'
    energy_steps, data = read_energy_data(filepath)
    print("Energy data read successfully.")
    log_scaled_data = build_log_scaled_dataset(energy_steps, data)
    log_scaled_data.to_csv('../data/log_scaled_flux_data.csv', index=False)
    energy_scaled_data = build_energy_scaled_dataset(energy_steps, data)
    energy_scaled_data.to_csv('../data/energy_scaled_flux_data.csv', index=False)
    print("Log-scaled and energy-scaled datasets created and saved.")