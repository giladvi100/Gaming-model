import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ORIGINALS_SYNTHETICS = [('WineQT', '../csvs/WineQT.csv', '../csvs/WineQT_vae_output.csv'),
                        ('Cleaned_data_houses', '../csvs/Cleaned_data_houses.csv', '../csvs/Cleaned_data_houses_gan_output.csv'),
                        ('Healthcare_Diabetes', '../csvs/Healthcare-Diabetes.csv', '../csvs/Healthcare-Diabetes_vae_output.csv'),
                        ('london_merged_smaller', '../csvs/Clean_london_merged_smaller.csv', '../csvs/Clean_london_merged_smaller_vae_output.csv')
                        ]

def plot_boxplot(csv_path, plt_name):
    try:
        data = pd.read_csv(csv_path)
        for column in data.select_dtypes(include=['float64', 'int64']).columns:
            if column not in ['quality', 'SalePrice', 'Outcome', 'cnt']:
                continue
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=data[column], color='blue')
            plt.title(f'Boxplot of {column} from data {plt_name}')
            plt.xlabel(column)
            plt.ylabel('Value Range')
            plt.grid(True)
            plt.savefig(f'./boxplots/{column}_{plt_name}', dpi=300)
    except FileNotFoundError:
        print(f"Error: File not found at path '{csv_path}'. Please provide a valid path.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage:
# Replace 'path_to_original_data.csv' and 'path_to_synthetic_data.csv' with actual file paths.
for data_name, original_p, synthetic_p in ORIGINALS_SYNTHETICS:
    plot_boxplot(original_p, f'{data_name}_boxplot_original')  # For original data
    plot_boxplot(synthetic_p, f'{data_name}_boxplot_synthetic')  # For synthetic data
