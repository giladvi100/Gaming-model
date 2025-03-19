import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from sklearn.preprocessing import StandardScaler

class VAEGenerator:
    def __init__(self, path):
        """
        Initialize the VAEGenerator with the path to the CSV dataset file.

        :param path: Path to the CSV dataset file.
        """
        self.path = path
        self.data = None
        self.data_scaled = None
        self.scaler = None
        self.input_dim = 6
        self.latent_dim = min(5, self.input_dim // 2)  # Dimensionality of latent space
        self.encoder = None
        self.decoder = None
        self.vae = None

    def load_data(self):
        """
        Load the dataset from the CSV file and normalize it.
        """
        self.data = pd.read_csv(self.path)
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(self.data)
        self.input_dim = self.data_scaled.shape[1]

    def build_vae(self, latent_dim, units_per_layer):
        """
        Build the VAE model.

        :param latent_dim: Dimension of the latent space.
        :param units_per_layer: Number of units in each layer.
        """
        # Encoder
        inputs = Input(shape=(self.input_dim,))
        x = Dense(units_per_layer, activation='relu')(inputs)
        x = Dense(units_per_layer // 2, activation='relu')(x)
        z_mean = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # Encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(units_per_layer // 2, activation='relu')(latent_inputs)
        x = Dense(units_per_layer, activation='relu')(x)
        outputs = Dense(self.input_dim, activation='linear')(x)

        # Decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')

        # VAE model
        vae_outputs = self.decoder(z)
        self.vae = Model(inputs, vae_outputs, name='vae')

    def compile_vae(self):
        """
        Compile the VAE model with a custom loss function.
        """
        def vae_loss(y_true, y_pred):
            reconstruction_loss = mse(y_true, y_pred)
            z_mean, z_log_var, _ = self.encoder(y_true)
            kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
            return reconstruction_loss + kl_loss

        self.vae.compile(optimizer='adam', loss=vae_loss)

    def train_vae(self, latent_dim, units_per_layer, epochs, batch_size):
        """
        Train the VAE model with specified hyperparameters.

        :param latent_dim: Dimension of the latent space.
        :param units_per_layer: Number of units in each layer.
        :param epochs: Number of epochs to train.
        :param batch_size: Batch size for training.
        """
        self.latent_dim = latent_dim
        self.input_dim = self.data_scaled.shape[1]

        self.build_vae(latent_dim, units_per_layer)
        self.compile_vae()
        self.vae.fit(self.data_scaled, self.data_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
        loss = self.vae.evaluate(self.data_scaled, self.data_scaled, verbose=0)
        return loss

    def grid_search(self):
        """
        Perform grid search to find the best hyperparameters.
        """
        param_grid = {
            'latent_dim': [2, 5, 10],
            'units_per_layer': [16, 32, 64],
            'epochs': [50, 100, 200],
            'batch_size': [32, 64, 128]
        }

        best_loss = float('inf')
        for latent_dim in param_grid['latent_dim']:
            for units_per_layer in param_grid['units_per_layer']:
                for epochs in param_grid['epochs']:
                    for batch_size in param_grid['batch_size']:
                        loss = self.train_vae(latent_dim, units_per_layer, epochs, batch_size)
                        if loss < best_loss:
                            best_loss = loss
                            self.best_params = {'latent_dim': latent_dim, 'units_per_layer': units_per_layer, 'epochs': epochs, 'batch_size': batch_size}
        print("Best Parameters:", self.best_params)
        return self.best_params

    def generate_samples(self, num_samples):
        """
        Generate new samples using the trained VAE model.

        :param num_samples: Number of samples to generate.
        :return: Generated samples.
        """
        random_latent_vectors = np.random.normal(size=(num_samples, self.latent_dim))
        generated_data = self.decoder.predict(random_latent_vectors)
        generated_data = self.scaler.inverse_transform(generated_data)
        return generated_data

    def save_samples(self, generated_data, filename):
        """
        Save generated samples to a CSV file.

        :param generated_data: Data to save.
        :param filename: Filename for the CSV file.
        """
        generated_df = pd.DataFrame(generated_data, columns=self.data.columns)
        generated_df.to_csv(filename, index=False)

    def generate(self, output_path, samples_number):
        self.load_data()
        best_params = self.grid_search()
        self.latent_dim = best_params['latent_dim']
        self.units_per_layer = best_params['units_per_layer']
        self.epochs = best_params['epochs']
        self.batch_size = best_params['batch_size']
        self.build_vae(self.latent_dim, self.units_per_layer)
        self.compile_vae()
        self.vae.fit(self.data_scaled, self.data_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        generated_data = self.generate_samples(samples_number)
        self.save_samples(generated_data, output_path)

# # Example usage
# if __name__ == "__main__":
#     vae_generator = VAEGenerator('../csvs/Clean_london_merged.csv')
#     vae_generator.generate('./csvs/wired_vae.csv')
