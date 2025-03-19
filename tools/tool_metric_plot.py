from GEN_DATA.statistical_evalator import StatisticalEvaluator

ORIGINALS_SYNTHETICS_GAN = [('WineQT', '../csvs/WineQT.csv', '../csvs/WineQT_gan_output.csv'),
                        ('Cleaned_data_houses', '../csvs/Cleaned_data_houses.csv', '../csvs/Cleaned_data_houses_gan_output.csv'),
                        ('Healthcare_Diabetes', '../csvs/Healthcare-Diabetes.csv', '../csvs/Healthcare-Diabetes_gan_output.csv'),
                        ('data_houses', '../csvs/Cleaned_data_houses.csv', '../csvs/Cleaned_data_houses_gan_output.csv'),
                        ('london_merged_smaller', '../csvs/Clean_london_merged_smaller.csv', '../csvs/Clean_london_merged_smaller_gan_output.csv')
                        ]
ORIGINALS_SYNTHETICS_VAE = [('WineQT', '../csvs/WineQT.csv', '../csvs/WineQT_vae_output.csv'),
                        ('Cleaned_data_houses', '../csvs/Cleaned_data_houses.csv', '../csvs/Cleaned_data_houses_vae_output.csv'),
                        ('Healthcare_Diabetes', '../csvs/Healthcare-Diabetes.csv', '../csvs/Healthcare-Diabetes_vae_output.csv'),
                        ('data_houses', '../csvs/Cleaned_data_houses.csv', '../csvs/Cleaned_data_houses_vae_output.csv'),
                        ('london_merged_smaller', '../csvs/Clean_london_merged_smaller.csv', '../csvs/Clean_london_merged_smaller_vae_output.csv')
                        ]

def plot_table(table_name, table_title, dcit_to_plot):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(dcit_to_plot)
    fig, ax = plt.subplots(figsize=(8, 2))  # Adjust size as needed
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    ax.axis('off')  # Turn off axis
    ax.axis('tight')  # Remove extra whitespace

    fig.suptitle(table_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to fit the title
    plt.savefig(table_name, dpi=300)  # Save as image
    plt.close()

results_list = []
for dataset_name, original, synthetic in ORIGINALS_SYNTHETICS_GAN:
    stat_evaluator_gan = StatisticalEvaluator(original, synthetic)
    results = stat_evaluator_gan.evaluate_similarity()
    results['dataset'] = dataset_name
    results_list.append(results)
    plot_table("metrics_gan", "GAN Model", results_list)

results_list = []
for dataset_name, original, synthetic in ORIGINALS_SYNTHETICS_VAE:
    stat_evaluator_gan = StatisticalEvaluator(original, synthetic)
    results = stat_evaluator_gan.evaluate_similarity()
    results['dataset'] = dataset_name
    results_list.append(results)
    plot_table("metrics_vae", "VAE Model", results_list)


results_list = [
    {
        'total samples': 600,
        'original samples': 600,
        'synthetic samples': 0,
        'R2 score': 0.5911,
        'MSE': 405337
    },
    {
        'total samples': 1200,
        'original samples': 1200,
        'synthetic samples': 0,
        'R2 score': 0.7213,
        'MSE': 374418
    },
    {
        'total samples': 1200,
        'original samples': 600,
        'synthetic samples': 600,
        'R2 score': 0.6611,
        'MSE': 311280
    }
]
plot_table("model_results","Model Results", results_list)