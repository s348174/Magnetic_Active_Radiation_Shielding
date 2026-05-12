import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_energy_data(file_path):
    """
    Reads energy data from a CSV file and returns a Numpy array and a DataFrame.
    """
    full_dataset = pd.read_csv(file_path)
    energy_steps = np.array(full_dataset.iloc[:, 0].values)
    data = full_dataset.iloc[:, 1:]
    return energy_steps, data

def build_log_scaled_dataset(energy_steps, data):
    """
    Builds a log-scaled dataset from the energy steps and data.
    Returns: a new DataFrame with log-scaled energy steps and data.
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
    
if __name__ == "__main__":
    filepath = '../data/flux_data_clean.csv'
    energy_steps, data = read_energy_data(filepath)
    print("Energy data read successfully.")
    log_scaled_data = build_log_scaled_dataset(energy_steps, data)
    log_scaled_data.to_csv('../data/log_scaled_flux_data.csv', index=False)
    print("Log-scaled dataset created and saved.")