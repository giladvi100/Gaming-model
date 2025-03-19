# This is a sample Python script.
import pandas as pd

from generative_models.gan_model import GANGenerator
from statistical_evalator import StatisticalEvaluator
from generative_models.vae_model import VAEGenerator
import os

TARGET_COLUMN = 'cnt'


if __name__ == '__main__':
    paths = ['./csvs/WineQT.csv',
             './csvs/Cleaned_data_houses.csv',
             './csvs/Healthcare-Diabetes.csv',
             './csvs/Clean_london_merged_smaller.csv'
             ]
    datasets = []
    for csv_path in paths:
        base_name, extension = os.path.splitext(csv_path)
        # GAN
        output_path_gan = f"{base_name}_{'gan_output'}{extension}"
        gan_generator = GANGenerator(csv_path)
        gan_generator.grid_search()
        gan_generator.generate(output_path_gan, 4000)
        print("GAN- Done")
        # VAE
        output_path_vae = f"{base_name}_{'vae_output'}{extension}"
        vae_generator = VAEGenerator(csv_path)
        vae_generator.generate(output_path_vae, 4000)
        print("VAE- Done")

        df_original_data = pd.read_csv(csv_path)
        df_vae_data = pd.read_csv(output_path_vae)
        df_gan_data = pd.read_csv(output_path_gan)
        concat_gan = pd.concat([df_original_data, df_gan_data])
        concat_vae = pd.concat([df_original_data, df_vae_data])
        concat_gan.to_csv(f"{base_name}_{'gan_concate'}{extension}")
        concat_vae.to_csv(f"{base_name}_{'vae_concate'}{extension}")

        datasets.append({
            "Original": csv_path,
            "GAN": output_path_gan,
            "VAE": output_path_vae,
            "Concate_GAN": './gan_concate.csv',
            "Concate_VAE": './vae_concate.csv',
        })

    #
    # print(datasets)
    for dataset_paths in datasets:
        dataset_paths_for_regression_model = {
            "Original": dataset_paths["Original"],
            "Concate_GAN": dataset_paths["Concate_GAN"],
            "Concate_VAE": dataset_paths["Concate_VAE"],
        }
    #     print(dataset_paths_for_regression_model)
    #     for key, dataset in dataset_paths_for_regression_model.items():
    #         print(f"Evaluating {key} dataset...")
    #         processor = DataProcessor(dataset, target_column=TARGET_COLUMN)
    #         X_train, X_test, y_train, y_test = processor.preprocess()
    #
    #         evaluator = ModelEvaluator(X_train, X_test, y_train, y_test)
    #         model_metrics = evaluator.train_and_evaluate()
    #
    #         print(f"Model Performance Metrics for {key}:")
    #         for metric, value in model_metrics.items():
    #             print(f"{metric}: {value:.4f}")
    #         print("-")

        import sys

        # Open a file for writing
        with open("output.txt", "w") as f:
            # Redirect print statements to both console and file
            def print_to_both(*args, **kwargs):
                print(*args, **kwargs)
                print(*args, **kwargs, file=f)


            print_to_both("Statistical Similarity Analysis:")

            print_to_both("GAN vs Original:", stat_evaluator_gan.evaluate_similarity())
            print_to_both("VAE vs Original:", stat_evaluator_vae.evaluate_similarity())

            stat_evaluator_gan = StatisticalEvaluator(dataset_paths["Original"], dataset_paths["GAN"], target_column=TARGET_COLUMN)
            stat_evaluator_vae = StatisticalEvaluator(dataset_paths["Original"], dataset_paths["VAE"], target_column=TARGET_COLUMN)

            weights = {
                "KL Divergence": 0.4,
                "Wasserstein Distance": 0.3,
                "Kolmogorov-Smirnov Test": 0.3
            }

            print_to_both(f"results for the dataset {dataset_paths["Original"]}")
            print_to_both("model 1 with gan, model 2 with vae")

            metrics_results_gan = stat_evaluator_gan.evaluate_similarity()
            metrics_results_vae = stat_evaluator_vae.evaluate_similarity()
            print_to_both(StatisticalEvaluator.weighted_comparison(metrics_results_gan, metrics_results_vae, weights))
            print_to_both("----------")


