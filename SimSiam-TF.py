# %%
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_augmentation_model():
    """Create data augmentation model for MNIST images."""
    return tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1)
    ])

def create_encoder():
    """Create encoder model for MNIST images."""
    return tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation=None)  # No activation in the final layer
    ])

def create_projector(input_dim=128, hidden_dim=128, output_dim=64):
    """Create projector model."""
    return tf.keras.Sequential([
        layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
        layers.Dense(output_dim, activation=None)
    ])

def create_predictor(input_dim=64, hidden_dim=64, output_dim=64):
    """Create predictor model."""
    return tf.keras.Sequential([
        layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
        layers.Dense(output_dim, activation=None)
    ])

class SimSiam(tf.keras.Model):
    def __init__(self):
        super(SimSiam, self).__init__()
        self.encoder = create_encoder()
        self.projector = create_projector()
        self.predictor = create_predictor()

    def call(self, inputs):
        z1 = self.projector(self.encoder(inputs[0]))
        z2 = self.projector(self.encoder(inputs[1]))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1, z2

@tf.function
def simsiam_loss(p1, p2, z1, z2):
    """Compute SimSiam loss."""
    # Stop gradient for z1 and z2
    z1 = tf.stop_gradient(z1)
    z2 = tf.stop_gradient(z2)

    # Normalize the projections and predictions
    p1 = tf.math.l2_normalize(p1, axis=1)
    p2 = tf.math.l2_normalize(p2, axis=1)
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    # Negative cosine similarity
    loss = -0.5 * (tf.reduce_mean(tf.reduce_sum(p1 * z2, axis=1)) +
                   tf.reduce_mean(tf.reduce_sum(p2 * z1, axis=1)))
    return loss

@tf.function
def train_step(model, optimizer, x1, x2):
    """Perform a single training step."""
    with tf.GradientTape() as tape:
        p1, p2, z1, z2 = model([x1, x2])
        loss = simsiam_loss(p1, p2, z1, z2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_simsiam(model, train_dataset, optimizer, epochs):
    """Train the SimSiam model."""
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for x1, x2 in train_dataset:
            loss = train_step(model, optimizer, x1, x2)
            total_loss += loss
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

def linear_evaluation(encoder, x_train, y_train, x_test, y_test):
    """Perform linear evaluation of the learned representations."""
    features_train = encoder.predict(x_train)
    features_test = encoder.predict(x_test)

    classifier = tf.keras.Sequential([
        layers.Dense(10, activation='softmax', input_shape=(128,))
    ])
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(features_train, y_train, epochs=100, batch_size=256, validation_split=0.2, verbose=0)

    _, accuracy = classifier.evaluate(features_test, y_test, verbose=0)
    return accuracy

def extract_features(encoder, data):
    """Extract features using the trained encoder."""
    return encoder.predict(data)

def plot_embeddings(features, labels, method='PCA'):
    """Plot embeddings using PCA or UMAP."""
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
    elif method == 'UMAP':
        reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    else:
        raise ValueError("Method must be either 'PCA' or 'UMAP'")
    
    embeddings = reducer.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'{method} visualization of SimSiam learned representations')
    plt.xlabel(f'{method}_1')
    plt.ylabel(f'{method}_2')
    plt.savefig(f'simsiam_mnist_{method.lower()}_tf.png')
    plt.close()

def main():
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Create augmentation model
    augmentation = create_augmentation_model()

    # Create dataset
    def augment_pair(image, _):
        return augmentation(image), augmentation(image)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).map(augment_pair).batch(256).prefetch(tf.data.AUTOTUNE)

    # Create and train SimSiam model
    simsiam_model = SimSiam()
    optimizer = optimizers.Adam(learning_rate=0.001)

    print("Training SimSiam model...")
    train_simsiam(simsiam_model, train_dataset, optimizer, epochs=50)

    # Linear evaluation
    print("Performing linear evaluation...")
    accuracy = linear_evaluation(simsiam_model.encoder, x_train, y_train, x_test, y_test)
    print(f"Linear evaluation accuracy: {accuracy:.4f}")

    # Extract features
    print("Extracting features...")
    train_features = extract_features(simsiam_model.encoder, x_train)
    test_features = extract_features(simsiam_model.encoder, x_test)

    # Visualize with PCA
    print("Generating PCA visualization...")
    plot_embeddings(test_features, y_test, method='PCA')

    # Visualize with UMAP
    print("Generating UMAP visualization...")
    plot_embeddings(test_features, y_test, method='UMAP')

if __name__ == "__main__":
    main()


