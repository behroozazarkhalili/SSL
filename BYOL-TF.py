# %%
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.decomposition import PCA
from umap import UMAP

# %%
class MNISTBYOLDataset:
    """
    Custom MNIST dataset for BYOL that returns two augmented versions of each image.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def augment(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random augmentations to the input image."""
        image = tf.image.random_crop(image, size=[28, 28, 1])
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image

    def __call__(self) -> tf.data.Dataset:
        """Create and return a tf.data.Dataset."""
        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        dataset = dataset.shuffle(10000)
        dataset = dataset.map(lambda x, y: (self.augment(x), self.augment(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

class EncoderNetwork(keras.Model):
    """
    Encoder network for BYOL.
    """
    def __init__(self, feature_dim: int = 256):
        super(EncoderNetwork, self).__init__()
        self.encoder = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(feature_dim)
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.encoder(x)

class ProjectionNetwork(keras.Model):
    """
    Projection network for BYOL.
    """
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, output_dim: int = 128):
        super(ProjectionNetwork, self).__init__()
        self.projection = keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(output_dim)
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.projection(x)

class PredictionNetwork(keras.Model):
    """
    Prediction network for BYOL.
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super(PredictionNetwork, self).__init__()
        self.prediction = keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(output_dim)
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.prediction(x)

class BYOL(keras.Model):
    """
    BYOL model combining encoder, projection, and prediction networks.
    """
    def __init__(self, feature_dim: int = 256, projection_dim: int = 128):
        super(BYOL, self).__init__()
        self.online_encoder = EncoderNetwork(feature_dim)
        self.online_projector = ProjectionNetwork(feature_dim, feature_dim, projection_dim)
        self.online_predictor = PredictionNetwork(projection_dim, feature_dim, projection_dim)
        
        self.target_encoder = EncoderNetwork(feature_dim)
        self.target_projector = ProjectionNetwork(feature_dim, feature_dim, projection_dim)
        
        # Build models
        dummy_input = tf.keras.Input(shape=(28, 28, 1))
        self.online_encoder(dummy_input)
        self.online_projector(self.online_encoder(dummy_input))
        self.online_predictor(self.online_projector(self.online_encoder(dummy_input)))
        self.target_encoder(dummy_input)
        self.target_projector(self.target_encoder(dummy_input))
        
        # Initialize target network with online network's parameters
        self.target_encoder.set_weights(self.online_encoder.get_weights())
        self.target_projector.set_weights(self.online_projector.get_weights())

    def call(self, x1: tf.Tensor, x2: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # Online network forward pass
        online_feat1 = self.online_encoder(x1)
        online_proj1 = self.online_projector(online_feat1)
        online_pred1 = self.online_predictor(online_proj1)
        
        online_feat2 = self.online_encoder(x2)
        online_proj2 = self.online_projector(online_feat2)
        online_pred2 = self.online_predictor(online_proj2)
        
        # Target network forward pass
        target_feat1 = self.target_encoder(x1)
        target_proj1 = self.target_projector(target_feat1)
        
        target_feat2 = self.target_encoder(x2)
        target_proj2 = self.target_projector(target_feat2)
        
        return online_pred1, online_pred2, target_proj1, target_proj2
@tf.function
def byol_loss(online_pred1: tf.Tensor, online_pred2: tf.Tensor, 
               target_proj1: tf.Tensor, target_proj2: tf.Tensor) -> tf.Tensor:
    """
    Compute BYOL loss.
    """
    loss1 = tf.reduce_mean(tf.losses.cosine_similarity(online_pred1, tf.stop_gradient(target_proj2), axis=-1))
    loss2 = tf.reduce_mean(tf.losses.cosine_similarity(online_pred2, tf.stop_gradient(target_proj1), axis=-1))
    return (loss1 + loss2) / 2

@tf.function
def train_step(model: BYOL, x1: tf.Tensor, x2: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer) -> tf.Tensor:
    """
    Perform a single training step.
    """
    with tf.GradientTape() as tape:
        online_pred1, online_pred2, target_proj1, target_proj2 = model(x1, x2)
        loss = byol_loss(online_pred1, online_pred2, target_proj1, target_proj2)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def update_moving_average(online_model: keras.Model, target_model: keras.Model, beta: float = 0.99):
    """
    Update moving average of target network.
    """
    for online_weights, target_weights in zip(online_model.get_weights(), target_model.get_weights()):
        target_weights = beta * target_weights + (1 - beta) * online_weights
        target_model.set_weights([target_weights if w.shape == target_weights.shape else w for w in target_model.get_weights()])

@tf.function
def train_step(model: BYOL, x1: tf.Tensor, x2: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer) -> tf.Tensor:
    """
    Perform a single training step.
    """
    with tf.GradientTape() as tape:
        online_pred1, online_pred2, target_proj1, target_proj2 = model(x1, x2)
        loss = byol_loss(online_pred1, online_pred2, target_proj1, target_proj2)
    
    trainable_vars = (model.online_encoder.trainable_variables + 
                      model.online_projector.trainable_variables + 
                      model.online_predictor.trainable_variables)
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss

def train_byol(model: BYOL, train_dataset: tf.data.Dataset, optimizer: tf.keras.optimizers.Optimizer, 
               epochs: int = 100) -> List[float]:
    """
    Train the BYOL model.
    """
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x1, x2, _) in enumerate(train_dataset):
            loss = train_step(model, x1, x2, optimizer)
            epoch_loss += loss.numpy()
            
            update_moving_average(model.online_encoder, model.target_encoder)
            update_moving_average(model.online_projector, model.target_projector)
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}], Loss: {loss.numpy():.4f}")
        
        avg_loss = epoch_loss / (batch_idx + 1)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
    
    return losses

class MNISTClassifier(keras.Model):
    """
    Classifier for MNIST using the pre-trained BYOL encoder.
    """
    def __init__(self, encoder: EncoderNetwork):
        super(MNISTClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = layers.Dense(10, activation='softmax')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        features = self.encoder(x)
        return self.classifier(features)

def train_classifier(model: MNISTClassifier, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, 
                     optimizer: tf.keras.optimizers.Optimizer, epochs: int = 10):
    """
    Fine-tune the classifier on MNIST.
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_acc_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def test_step(x, y):
        logits = model(x, training=False)
        test_acc_metric.update_state(y, logits)

    for epoch in range(epochs):
        for x1, _, y in train_dataset:  # Use only the first augmented view
            loss_value = train_step(x1, y)

        train_acc = train_acc_metric.result()
        print(f"Epoch {epoch + 1}, Loss: {loss_value.numpy():.4f}, Accuracy: {train_acc.numpy():.4f}")

        for x, y in test_dataset:
            test_step(x, y)

        test_acc = test_acc_metric.result()
        print(f"Test Accuracy: {test_acc.numpy():.4f}")

        train_acc_metric.reset_states()
        test_acc_metric.reset_states()

def extract_features(model: keras.Model, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features using the trained encoder.
    """
    features = []
    labels = []
    for x, y in dataset:
        feature = model.encoder(x, training=False)
        features.append(feature.numpy())
        labels.append(y.numpy())
    return np.vstack(features), np.concatenate(labels)

def plot_embeddings(features: np.ndarray, labels: np.ndarray, method: str = 'UMAP'):
    """
    Plot embeddings using UMAP or PCA.
    """
    if method == 'UMAP':
        reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    elif method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be either 'UMAP' or 'PCA'")
    
    embeddings = reducer.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'{method} visualization of BYOL learned representations')
    plt.xlabel(f'{method}_1')
    plt.ylabel(f'{method}_2')
    plt.savefig(f'byol_mnist_{method.lower()}_tf.png')
    plt.close()


# %%

def main():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

    # Create datasets
    train_dataset = MNISTBYOLDataset(x_train, y_train, batch_size=256)()
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

    # Initialize BYOL model
    byol_model = BYOL()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # Train BYOL
    print("Starting BYOL training...")
    losses = train_byol(byol_model, train_dataset, optimizer, epochs=10)

    # Plot BYOL training loss
    plt.plot(losses)
    plt.title('BYOL Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('byol_training_loss_tf.png')
    plt.close()

    # Initialize and train classifier
    classifier = MNISTClassifier(byol_model.online_encoder)
    classifier_optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    print("Starting classifier fine-tuning...")
    train_classifier(classifier, train_dataset, test_dataset, classifier_optimizer, epochs=10)

    # Extract features and create visualizations
    features, labels = extract_features(classifier, test_dataset)

    print("Generating UMAP visualization...")
    plot_embeddings(features, labels, method='UMAP')

    print("Generating PCA visualization...")
    plot_embeddings(features, labels, method='PCA')

    print("BYOL training and evaluation complete!")

if __name__ == "__main__":
    main()


