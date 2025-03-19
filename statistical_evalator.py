import pandas as pd
import numpy as np
from scipy.stats import entropy, wasserstein_distance, ks_2samp


class StatisticalEvaluator:
    def __init__(self, original_file, synthetic_file, target_column='Critic_Score'):
        self.original_data = pd.read_csv(original_file)#.drop(columns=[target_column])
        self.synthetic_data = pd.read_csv(synthetic_file)#.drop(columns=[target_column])

    def kl_divergence(self):
        kl_scores = []
        for col in self.original_data.columns:
            p = np.histogram(self.original_data[col], bins=30, density=True)[0] + 1e-10
            q = np.histogram(self.synthetic_data[col], bins=30, density=True)[0] + 1e-10
            kl_scores.append(entropy(p, q))
        return np.mean(kl_scores)

    def wasserstein_distance_metric(self):
        return np.mean([wasserstein_distance(self.original_data[col], self.synthetic_data[col]) for col in
                        self.original_data.columns])

    def ks_test(self):
        return np.mean(
            [ks_2samp(self.original_data[col], self.synthetic_data[col])[0] for col in self.original_data.columns])

    def evaluate_similarity(self):
        return {
            "KL Divergence": self.kl_divergence(),
            "Wasserstein Distance": self.wasserstein_distance_metric(),
            "Kolmogorov-Smirnov Test": self.ks_test()
        }

    @staticmethod
    def weighted_comparison(model1_results, model2_results, weights):
        """
        Compare two models using weighted metrics.

        Parameters:
        - model1_results: Dictionary of metrics for the first model.
        - model2_results: Dictionary of metrics for the second model.
        - weights: Dictionary with weights for each metric.

        Returns:
        - A string indicating which model performed better.
        """

        metrics = ["KL Divergence", "Wasserstein Distance", "Kolmogorov-Smirnov Test"]

        model1_score = sum(weights[metric] * model1_results[metric] for metric in metrics)
        model2_score = sum(weights[metric] * model2_results[metric] for metric in metrics)

        if model1_score < model2_score:
            return "Model 1 performed better."
        elif model2_score < model1_score:
            return "Model 2 performed better."
        else:
            return "Both models performed equally well."

