import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

class GANGenerator:
    def __init__(self, csv_path):
        """
        Initialize the GAN generator with the path to the CSV file.

        :param csv_path: Path to the CSV file containing numerical data.
        """
        self.csv_path = csv_path
        self.latent_dim = 128
        self.batch_size = 32
        self.epochs = 500
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()

    def load_data(self):
        """
        Load the data from the CSV file.
        """
        self.data = pd.read_csv(self.csv_path)

    def preprocess_data(self):
        """
        Preprocess the data by dropping missing values and scaling.
        """
        # Drop rows with missing values
        self.data = self.data.dropna()

        # Select numerical features
        X = self.data.values

        # Split the data
        self.X_train, self.X_test = train_test_split(X, test_size=0.2, random_state=42)

        # Scale the data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def build_generator(self, latent_dim, units_per_layer):
        """
        Build the generator model.

        :param latent_dim: Dimension of the latent space.
        :param units_per_layer: Number of units in each layer.
        :return: The generator model.
        """
        model = Sequential([
            Dense(units_per_layer, input_dim=latent_dim),
            LeakyReLU(0.2),
            BatchNormalization(),
            Dense(units_per_layer * 2),
            LeakyReLU(0.2),
            BatchNormalization(),
            Dense(self.X_train.shape[1], activation='tanh')
        ])
        return model

    def build_discriminator(self, units_per_layer):
        """
        Build the discriminator model.

        :param units_per_layer: Number of units in each layer.
        :return: The discriminator model.
        """
        model = Sequential([
            Dense(units_per_layer * 2, input_dim=self.X_train.shape[1]),
            LeakyReLU(0.2),
            Dropout(0.3),
            Dense(units_per_layer),
            LeakyReLU(0.2),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        return model

    def train_gan(self, latent_dim, units_per_layer, batch_size, epochs):
        """
        Train the GAN model with specified hyperparameters.

        :param latent_dim: Dimension of the latent space.
        :param units_per_layer: Number of units in each layer.
        :param batch_size: Batch size for training.
        :param epochs: Number of epochs to train.
        """
        self.generator = self.build_generator(latent_dim, units_per_layer)
        self.discriminator = self.build_discriminator(units_per_layer)

        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

        self.gan = Sequential([self.generator, self.discriminator])
        self.discriminator.trainable = False
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

        best_g_loss = float('inf')
        for epoch in range(epochs):
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            generated_data = self.generator.predict(noise)
            real_data = self.X_train_scaled[np.random.randint(0, self.X_train_scaled.shape[0], batch_size)]

            d_loss_real = self.discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

            if g_loss < best_g_loss:
                best_g_loss = g_loss
                self.best_params = {'latent_dim': latent_dim, 'units_per_layer': units_per_layer, 'batch_size': batch_size, 'epochs': epochs}

    def grid_search(self):
        """
        Perform grid search to find the best hyperparameters.
        """
        param_grid = {
            'latent_dim': [64, 128, 256],
            'units_per_layer': [32, 64, 128],
            'batch_size': [16, 32, 64],
            'epochs': [500, 1000]
        }

        best_g_loss = float('inf')
        for latent_dim in param_grid['latent_dim']:
            for units_per_layer in param_grid['units_per_layer']:
                for batch_size in param_grid['batch_size']:
                    for epochs in param_grid['epochs']:
                        self.train_gan(latent_dim, units_per_layer, batch_size, epochs)
                        if hasattr(self, 'best_params'):
                            if self.best_params['latent_dim'] == latent_dim and self.best_params['units_per_layer'] == units_per_layer and self.best_params['batch_size'] == batch_size and self.best_params['epochs'] == epochs:
                                print(f"Best Parameters Found: {self.best_params}")
                                self.latent_dim = self.best_params['latent_dim']
                                self.units_per_layer = self.best_params['units_per_layer']
                                self.batch_size = self.best_params['batch_size']
                                self.epochs = self.best_params['epochs']
                                return self.best_params

    def generate_data(self, num_samples):
        """
        Generate new data using the trained GAN model.

        :param num_samples: Number of samples to generate.
        :return: The generated data.
        """
        gan_generated = self.generator.predict(np.random.normal(0, 1, (num_samples, self.latent_dim)))
        gan_generated = self.scaler.inverse_transform(gan_generated)
        return gan_generated

    def save_generated_data(self, num_samples, output_path):
        """
        Save the generated data to a CSV file.

        :param num_samples: Number of samples to generate.
        :param output_path: Path to save the generated data.
        """
        generated_data = self.generate_data(num_samples)
        generated_df = pd.DataFrame(generated_data, columns=self.data.columns)
        generated_df.to_csv(output_path, index=False)

    def generate(self, output_path, samples_number):
        self.train_gan(self.latent_dim, 64, self.batch_size, self.epochs)
        self.save_generated_data(samples_number, output_path)

# # Example usage
# if __name__ == "__main__":
#     csv_path = '../csvs/Clean_london_merged.csv'
#     output_path = '../csvs/wired.csv'
#
#     gan_generator = GANGenerator(csv_path)
#     best_params = gan_generator.grid_search()
#     print("Best Parameters:", best_params)
#
#     gan_generator.generate(output_path)
