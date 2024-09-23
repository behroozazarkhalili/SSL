# %%
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP

# 1. Data loading and augmentation
def create_augmented_dataset(x, y, batch_size):
    def augment(image):
        image = tf.image.random_crop(image, size=[28, 28, 1])
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image

    def create_pair(image, label):
        return augment(image), augment(image), label

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(create_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# 2. Barlow Twins model architecture
class BarlowTwinsMNIST(keras.Model):
    def __init__(self, feature_dim=128):
        super(BarlowTwinsMNIST, self).__init__()
        self.encoder = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(feature_dim)
        ])
        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        z = self.encoder(inputs)
        z = self.bn(z)
        return z

# 3. Barlow Twins loss function
@tf.function
def barlow_twins_loss(z1, z2, lambda_param=5e-3):
    batch_size = tf.shape(z1)[0]
    feature_dim = tf.shape(z1)[1]
    
    c = tf.matmul(z1, z2, transpose_a=True) / tf.cast(batch_size, tf.float32)
    c_diff = (c - tf.eye(feature_dim)) ** 2
    
    off_diagonal = tf.linalg.set_diag(c_diff, tf.zeros(feature_dim))
    loss = tf.reduce_sum(tf.linalg.diag_part(c_diff)) + lambda_param * tf.reduce_sum(off_diagonal)
    
    return loss

# 4. Pre-training function
@tf.function
def train_step(model, optimizer, x1, x2):
    with tf.GradientTape() as tape:
        z1 = model(x1)
        z2 = model(x2)
        loss = barlow_twins_loss(z1, z2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def pretrain_barlow_twins(model, train_dataset, optimizer, epochs=10):
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for x1, x2, _ in train_dataset:
            loss = train_step(model, optimizer, x1, x2)
            epoch_loss += loss
        avg_loss = epoch_loss / len(train_dataset)
        losses.append(avg_loss.numpy())
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    return losses

# 5. Classifier for fine-tuning
class MNISTClassifier(keras.Model):
    def __init__(self, encoder):
        super(MNISTClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        features = self.encoder(inputs)
        return self.classifier(features)

# 6. Fine-tuning function
@tf.function
def train_classifier_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def finetune(model, train_dataset, test_dataset, optimizer, epochs=10):
    for epoch in range(epochs):
        for x, y in train_dataset:
            loss = train_classifier_step(model, optimizer, x, y)
        
        # Evaluate on test set
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        for x, y in test_dataset:
            y_pred = model(x)
            test_accuracy.update_state(y, y_pred)
        print(f'Epoch {epoch+1}, Test Accuracy: {test_accuracy.result().numpy():.4f}')

# 7. Feature extraction function
def extract_features(model, dataset):
    features = []
    labels = []
    for x, y in dataset:
        feature = model.encoder(x)
        features.append(feature.numpy())
        labels.append(y.numpy())
    return np.vstack(features), np.concatenate(labels)

# 8. Visualization function
def plot_embeddings(features, labels, method='UMAP'):
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
    plt.title(f'{method} visualization of Barlow Twins learned representations')
    plt.xlabel(f'{method}_1')
    plt.ylabel(f'{method}_2')
    plt.savefig(f'barlow_twins_mnist_{method.lower()}_tf.png')
    plt.close()

# 9. Main execution
def main():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

    # Create datasets
    batch_size = 256
    train_dataset_pretraining = create_augmented_dataset(x_train, y_train, batch_size)
    
    # Create separate datasets for fine-tuning
    train_dataset_finetuning = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # Initialize model and optimizer for pre-training
    model = BarlowTwinsMNIST()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # Pre-train the model
    print("Starting pre-training...")
    pretrain_losses = pretrain_barlow_twins(model, train_dataset_pretraining, optimizer, epochs=50)

    # Plot pre-training loss
    plt.plot(pretrain_losses)
    plt.title('Barlow Twins Pre-training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('barlow_twins_pretrain_loss_tf.png')
    plt.close()

    # Save the pre-trained model
    model.save_weights('barlow_twins_pretrained_mnist_tf.h5')
    print("Pre-training complete!")

    # Initialize classifier for fine-tuning
    classifier = MNISTClassifier(model.encoder)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # Fine-tune the model
    print("Starting fine-tuning...")
    finetune(classifier, train_dataset_finetuning, test_dataset, optimizer, epochs=10)

    print("Fine-tuning complete!")

    # Extract features for visualization
    features, labels = extract_features(classifier, test_dataset)

    # Plot UMAP
    print("Generating UMAP visualization...")
    plot_embeddings(features, labels, method='UMAP')

    # Plot PCA
    print("Generating PCA visualization...")
    plot_embeddings(features, labels, method='PCA')

    print("Visualizations complete!")

if __name__ == "__main__":
    main()


