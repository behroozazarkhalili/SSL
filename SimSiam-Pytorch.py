# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

class MNISTSimSiamDataset(Dataset):
    """
    Custom MNIST dataset for SimSiam that returns two augmented versions of each image.
    """
    def __init__(self, root: str, train: bool = True, download: bool = False):
        self.mnist = datasets.MNIST(root, train=train, download=download)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img, target = self.mnist[index]
        return self.transform(img), self.transform(img), target

    def __len__(self) -> int:
        return len(self.mnist)

class SimSiamMNIST(nn.Module):
    """
    SimSiam model for MNIST.
    """
    def __init__(self, feature_dim: int = 256, pred_dim: int = 128):
        super(SimSiamMNIST, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(9216, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
        # Projection MLP
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Prediction MLP
        self.prediction = nn.Sequential(
            nn.Linear(feature_dim, pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, feature_dim)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.projection(self.encoder(x1))
        z2 = self.projection(self.encoder(x2))
        
        p1 = self.prediction(z1)
        p2 = self.prediction(z2)
        
        return p1, p2, z1.detach(), z2.detach()

def simsiam_loss(p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Compute SimSiam loss.
    """
    # Negative cosine similarity
    return - (nn.functional.cosine_similarity(p1, z2.detach(), dim=-1).mean() + 
              nn.functional.cosine_similarity(p2, z1.detach(), dim=-1).mean()) * 0.5

def train_simsiam(model: SimSiamMNIST, train_loader: DataLoader, optimizer: optim.Optimizer, 
                  device: torch.device, epochs: int = 100) -> List[float]:
    """
    Train the SimSiam model.
    """
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x1, x2, _) in enumerate(train_loader):
            x1, x2 = x1.to(device), x2.to(device)
            
            optimizer.zero_grad()
            p1, p2, z1, z2 = model(x1, x2)
            loss = simsiam_loss(p1, p2, z1, z2)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
    
    return losses

class MNISTClassifier(nn.Module):
    """
    Classifier for MNIST using the pre-trained SimSiam encoder.
    """
    def __init__(self, encoder: nn.Module):
        super(MNISTClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(256, 10)  # 10 classes for MNIST

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)

def train_classifier(model: MNISTClassifier, train_loader: DataLoader, test_loader: DataLoader, 
                     optimizer: optim.Optimizer, device: torch.device, epochs: int = 10):
    """
    Fine-tune the classifier on MNIST.
    """
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, _, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Test Accuracy: {accuracy:.2f}%")
        model.train()

def extract_features(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features using the trained encoder.
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            feature = model.encoder(data)
            features.append(feature.cpu().numpy())
            labels.append(target.numpy())
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
    plt.title(f'{method} visualization of SimSiam learned representations')
    plt.xlabel(f'{method}_1')
    plt.ylabel(f'{method}_2')
    plt.savefig(f'simsiam_mnist_{method.lower()}.png')
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets and data loaders
    train_dataset = MNISTSimSiamDataset('data', train=True, download=True)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    # Initialize SimSiam model
    simsiam_model = SimSiamMNIST().to(device)
    optimizer = optim.Adam(simsiam_model.parameters(), lr=0.001)
    
    # Train SimSiam
    print("Starting SimSiam training...")
    losses = train_simsiam(simsiam_model, train_loader, optimizer, device, epochs=10)
    
    # Plot SimSiam training loss
    plt.plot(losses)
    plt.title('SimSiam Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('simsiam_training_loss.png')
    plt.close()
    
    # Initialize and train classifier
    classifier = MNISTClassifier(simsiam_model.encoder).to(device)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    print("Starting classifier fine-tuning...")
    train_classifier(classifier, train_loader, test_loader, classifier_optimizer, device, epochs=50)
    
    # Extract features and create visualizations
    features, labels = extract_features(classifier, test_loader, device)
    
    print("Generating UMAP visualization...")
    plot_embeddings(features, labels, method='UMAP')
    
    print("Generating PCA visualization...")
    plot_embeddings(features, labels, method='PCA')
    
    print("SimSiam training and evaluation complete!")

if __name__ == "__main__":
    main()


