import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class Config:
    def __init__(self):
        self.batch_size = 256
        self.feature_dim = 256
        self.pred_dim = 256
        self.learning_rate = 0.05
        self.momentum = 0.99
        self.weight_decay = 1e-4
        self.epochs = 50

def create_encoder(feature_dim):
    return keras.Sequential([
        layers.Conv2D(64, 3, 1, 'same', activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, 1, 'same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(256, 3, 1, 'same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(feature_dim)
    ])

def create_predictor(feature_dim, pred_dim):
    return keras.Sequential([
        layers.Dense(pred_dim, activation='relu', input_shape=(feature_dim,)),
        layers.BatchNormalization(),
        layers.Dense(feature_dim)
    ])

class DirectPred(keras.Model):
    def __init__(self, feature_dim, pred_dim):
        super(DirectPred, self).__init__()
        self.online_encoder = create_encoder(feature_dim)
        self.predictor = create_predictor(feature_dim, pred_dim)
        self.target_encoder = create_encoder(feature_dim)
        
        # Initialize target_encoder with online_encoder's weights
        self.target_encoder.set_weights(self.online_encoder.get_weights())

    def call(self, inputs):
        x1, x2 = inputs
        z1 = self.online_encoder(x1)
        z2 = self.online_encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        t1 = tf.stop_gradient(self.target_encoder(x1))
        t2 = tf.stop_gradient(self.target_encoder(x2))
        return p1, p2, t1, t2

@tf.function
def directpred_loss(p1, p2, t1, t2):
    loss = tf.reduce_mean(tf.keras.losses.MSE(p1, t2) + tf.keras.losses.MSE(p2, t1))
    return loss

def update_target_encoder(online_encoder, target_encoder, momentum):
    for online_weights, target_weights in zip(online_encoder.weights, target_encoder.weights):
        target_weights.assign(momentum * target_weights + (1 - momentum) * online_weights)

@tf.function
def train_step(model, optimizer, x1, x2):
    with tf.GradientTape() as tape:
        p1, p2, t1, t2 = model((x1, x2))
        loss = directpred_loss(p1, p2, t1, t2)
    
    gradients = tape.gradient(loss, model.online_encoder.trainable_variables + model.predictor.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.online_encoder.trainable_variables + model.predictor.trainable_variables))
    
    update_target_encoder(model.online_encoder, model.target_encoder, Config().momentum)
    
    return loss

def prepare_dataset(images, labels, is_train=True):
    def prepare_sample(image, label):
        if is_train:
            image = tf.image.random_crop(image, size=[28, 28, 1])
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image1 = tf.image.random_flip_left_right(image)
            image2 = tf.image.random_flip_left_right(image)
            return (image1, image2), label
        else:
            return image, label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(prepare_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(Config().batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs, learning_rate):
    if epoch < warmup_epochs:
        return learning_rate * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return learning_rate * 0.5 * (1 + np.cos(np.pi * progress))

def prepare_dataset(images, labels, is_train=True):
    def prepare_sample(image, label):
        if is_train:
            image1 = tf.image.random_crop(image, size=[28, 28, 1])
            image1 = tf.image.random_flip_left_right(image1)
            image2 = tf.image.random_crop(image, size=[28, 28, 1])
            image2 = tf.image.random_flip_left_right(image2)
            return (image1, image2), label
        else:
            return image, label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(prepare_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(Config().batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train(model, train_dataset, test_dataset, optimizer, config):
    for epoch in range(config.epochs):
        total_loss = 0
        for (x1, x2), _ in train_dataset:
            loss = train_step(model, optimizer, x1, x2)
            total_loss += loss
        
        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            linear_acc, knn_acc = evaluate(model, train_dataset, test_dataset)
            print(f"Linear Accuracy: {linear_acc:.4f}, KNN Accuracy: {knn_acc:.4f}")

def extract_features(model, dataset):
    features, labels = [], []
    for x, y in dataset:
        if isinstance(x, tuple):
            x = x[0]  # Use only the first augmented view for feature extraction
        feature = model.online_encoder(x, training=False)
        features.append(feature.numpy())
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)

def evaluate(model, train_dataset, test_dataset):
    train_features, train_labels = extract_features(model, train_dataset)
    test_features, test_labels = extract_features(model, test_dataset)

    # Linear evaluation
    linear_model = keras.Sequential([
        keras.layers.Dense(10, activation='softmax')
    ])
    linear_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    linear_model.fit(train_features, train_labels, epochs=100, verbose=1)
    _, linear_acc = linear_model.evaluate(test_features, test_labels, verbose=1)

    # KNN evaluation
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_features, train_labels)
    knn_pred = knn.predict(test_features)
    knn_acc = accuracy_score(test_labels, knn_pred)

    return linear_acc, knn_acc


def main():
    config = Config()

    # Load and prepare data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

    train_dataset = prepare_dataset(x_train, y_train)
    test_dataset = prepare_dataset(x_test, y_test, is_train=False)

    # Create model and optimizer
    model = DirectPred(config.feature_dim, config.pred_dim)
    optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=0.9)

    # Train the model
    print("Starting DirectPred training...")
    train(model, train_dataset, test_dataset, optimizer, config)

    # Final evaluation
    print("\nFinal Evaluation:")
    linear_acc, knn_acc = evaluate(model, train_dataset, test_dataset)
    print(f"Linear Accuracy: {linear_acc:.4f}")
    print(f"KNN Accuracy: {knn_acc:.4f}")

if __name__ == "__main__":
    main()