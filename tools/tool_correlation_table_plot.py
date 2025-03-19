import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ORIGINALS_SYNTHETICS = [('WineQT', '../csvs/WineQT.csv', '../csvs/WineQT_vae_output.csv'),
                        ('Cleaned_data_houses', '../csvs/Cleaned_data_houses.csv', '../csvs/Cleaned_data_houses_gan_output.csv'),
                        ('Healthcare_Diabetes', '../csvs/Healthcare-Diabetes.csv', '../csvs/Healthcare-Diabetes_vae_output.csv'),
                        ('london_merged_smaller', '../csvs/Clean_london_merged_smaller.csv', '../csvs/Clean_london_merged_smaller_vae_output.csv')
                        ]
def plot_and_save_correlation_matrix(csv_path, save_folder, filename):
    """
    Function to generate and save the correlation matrix plot for a given CSV file.

    Parameters:
    csv_path (str): Path to the CSV file.
    save_folder (str): Path to the folder where the plot will be saved.
    filename (str): Name of the file to save the plot as (e.g., 'correlation_matrix.png').

    Returns:
    None
    """
    try:
        # Load the data
        data = pd.read_csv(csv_path)

        # Calculate the correlation matrix
        corr = data.corr()

        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)

        # Create the plot
        plt.figure(figsize=(12, 10))  # Adjust the figure size for better readability
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of All Columns')

        # Save the plot to the specified folder
        save_path = os.path.join(save_folder, filename)
        plt.savefig(save_path, bbox_inches="tight")  # Use bbox_inches to prevent labels from being cut off

        # Close the plot to free memory
        plt.close()

    except FileNotFoundError:
        print(f"Error: File not found at path '{csv_path}'. Please provide a valid path.")
    except ValueError as ve:
        print(f"ValueError: {ve}. Check if the CSV contains numeric data.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# Replace 'path_to_original_data.csv' and 'path_to_synthetic_data.csv' with actual file paths.
for data_name, original_p, synthetic_p in ORIGINALS_SYNTHETICS:
    plot_and_save_correlation_matrix(original_p, 'correlation', f'{data_name}_boxplot_original')  # For original data
    plot_and_save_correlation_matrix(synthetic_p, 'correlation',f'{data_name}_boxplot_synthetic')  # For synthetic data
