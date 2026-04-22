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
    # Map log_energy_steps in the same range as the original energy steps
    log_energy_steps = (log_energy_steps - log_energy_steps[0]) / (log_energy_steps[-1] - log_energy_steps[0]) * (energy_steps[-1] - energy_steps[0]) + energy_steps[0]
    #step_size = (log_energy_steps[-1] - log_energy_steps[0]) / (log_energy_steps.shape[0] - 1)
    #print(f"Log energy steps: {log_energy_steps}")
    #print(f"Start log energy step: {log_energy_steps[0]:.4f}, End log energy step: {log_energy_steps[-1]:.4f}")
    step_sizes = np.diff(log_energy_steps)
    step_size = np.mean(step_sizes)
    print(f"Step size for log energy steps: {step_size:.4f}")
    #print(f"Step sizes for log energy steps: {step_sizes}")
    normalized_data = data.copy()
    for column in data.columns:
        area = 0
        for i in range(log_energy_steps.shape[0]):
            #step_size = step_sizes[i] if i < step_sizes.shape[0] else step_sizes[-1]  # Use the last step size for the last point
            area += step_size * data[column].iloc[i]
        normalized_data[column] = data[column] / area if area != 0 else data[column]
    log_scaled_dataset = pd.DataFrame(normalized_data, columns=data.columns)
    log_scaled_dataset.insert(0, 'log_energy_steps', log_energy_steps)
    return log_scaled_dataset

def plot_log_scaled_data(log_scaled_dataset):
    """
    Plots the log-scaled dataset.
    
    Parameters:
    log_scaled_dataset (pd.DataFrame): The input DataFrame containing log-scaled energy data.
    """
    plt.figure(figsize=(10, 6))
    for column in log_scaled_dataset.columns[1:]:  # Skip the first column (log_energy_steps)
        plt.plot(log_scaled_dataset['log_energy_steps'], log_scaled_dataset[column], label=column)
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
        num = np.trapezoid(log_scaled_dataset['log_energy_steps'].values*log_scaled_dataset[column].values, log_scaled_dataset['log_energy_steps'].values)
        denom = np.trapezoid(log_scaled_dataset[column].values, log_scaled_dataset['log_energy_steps'].values)
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
        num = np.trapezoid((log_scaled_dataset['log_energy_steps'].values)**2 * log_scaled_dataset[column].values, log_scaled_dataset['log_energy_steps'].values)
        denom = np.trapezoid(log_scaled_dataset[column].values, log_scaled_dataset['log_energy_steps'].values)
        mean_sq = num / denom if denom != 0 else 0
        variances.append(mean_sq - mean**2)
    return np.array(variances)

def q_q_plot(log_scaled_dataset):
    """
    Generates a Q-Q plot for each column in the log-scaled dataset to visually assess normality.
    
    Parameters:
    log_scaled_dataset (pd.DataFrame): The input DataFrame containing log-scaled energy data.
    """
    for column in log_scaled_dataset.columns[1:]:  # Skip the first column (log_energy_steps)
        plt.figure(figsize=(6, 6))
        stats.probplot(log_scaled_dataset[column], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot for {column}')
        plt.grid()
        plt.show()

def test_normality(log_scaled_data):
    """
    Fits a log normal distribution to each column of the log-scaled dataset using empirical params."""
    means = compute_column_means(log_scaled_data)
    variances = compute_column_variances(log_scaled_data)
    stds = np.sqrt(variances)

    for column in log_scaled_data.columns[1:]:  # Skip the first column (log_energy_steps)
        # Test if the points lie on a normal distribution
        mean = means[log_scaled_data.columns.get_loc(column) - 1]  # Get the mean for the current column
        std = stds[log_scaled_data.columns.get_loc(column) - 1] 
        # KS test
        stat_ks, pval_ks = stats.kstest(log_scaled_data[column], stats.norm.cdf, args=(mean, std))
        # D'Agostino-Pearson test
        stat_dp, pval_dp = stats.normaltest(log_scaled_data[column])
        print(f"Column: {column}, KS Statistic: {stat_ks:.4f}, P-Value: {pval_ks:.4f}")
        print(f"Column: {column}, D'Agostino-Pearson Statistic: {stat_dp:.4f}, P-Value: {pval_dp:.4f}")

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
    plot_log_scaled_data(log_scaled_data)
    test_normality(log_scaled_data)
    #q_q_plot(log_scaled_data)

if __name__ == "__main__":
    test_stats()