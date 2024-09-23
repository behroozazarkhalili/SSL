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

# %%
class MNISTBYOLDataset(Dataset):
    """
    Custom MNIST dataset for BYOL that returns two augmented versions of each image.
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

class EncoderNetwork(nn.Module):
    """
    Encoder network for BYOL.
    """
    def __init__(self, feature_dim: int = 256):
        super(EncoderNetwork, self).__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class ProjectionNetwork(nn.Module):
    """
    Projection network for BYOL.
    """
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, output_dim: int = 128):
        super(ProjectionNetwork, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class PredictionNetwork(nn.Module):
    """
    Prediction network for BYOL.
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super(PredictionNetwork, self).__init__()
        self.prediction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prediction(x)

class BYOL(nn.Module):
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
        
        # Initialize target network with online network's parameters
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        
        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Online network forward pass
        online_feat1 = self.online_encoder(x1)
        online_proj1 = self.online_projector(online_feat1)
        online_pred1 = self.online_predictor(online_proj1)
        
        online_feat2 = self.online_encoder(x2)
        online_proj2 = self.online_projector(online_feat2)
        online_pred2 = self.online_predictor(online_proj2)
        
        # Target network forward pass
        with torch.no_grad():
            target_feat1 = self.target_encoder(x1)
            target_proj1 = self.target_projector(target_feat1)
            
            target_feat2 = self.target_encoder(x2)
            target_proj2 = self.target_projector(target_feat2)
        
        return online_pred1, online_pred2, target_proj1.detach(), target_proj2.detach()

def byol_loss(online_pred1: torch.Tensor, online_pred2: torch.Tensor, 
               target_proj1: torch.Tensor, target_proj2: torch.Tensor) -> torch.Tensor:
    """
    Compute BYOL loss.
    """
    loss1 = 2 - 2 * (online_pred1 * target_proj2).sum(dim=-1).mean()
    loss2 = 2 - 2 * (online_pred2 * target_proj1).sum(dim=-1).mean()
    return (loss1 + loss2) / 2

def update_moving_average(online_model: nn.Module, target_model: nn.Module, beta: float = 0.99):
    """
    Update moving average of target network.
    """
    for online_params, target_params in zip(online_model.parameters(), target_model.parameters()):
        target_params.data = beta * target_params.data + (1 - beta) * online_params.data

def train_byol(model: BYOL, train_loader: DataLoader, optimizer: optim.Optimizer, 
               device: torch.device, epochs: int = 100) -> List[float]:
    """
    Train the BYOL model.
    """
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x1, x2, _) in enumerate(train_loader):
            x1, x2 = x1.to(device), x2.to(device)
            online_pred1, online_pred2, target_proj1, target_proj2 = model(x1, x2)
            loss = byol_loss(online_pred1, online_pred2, target_proj1, target_proj2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            update_moving_average(model.online_encoder, model.target_encoder)
            update_moving_average(model.online_projector, model.target_projector)
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
    
    return losses

class MNISTClassifier(nn.Module):
    """
    Classifier for MNIST using the pre-trained BYOL encoder.
    """
    def __init__(self, encoder: EncoderNetwork):
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
        for batch_idx, (data1, data2, target) in enumerate(train_loader):
            # Use only the first augmented view (data1) for classification
            data, target = data1.to(device), target.to(device)
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
    plt.title(f'{method} visualization of BYOL learned representations')
    plt.xlabel(f'{method}_1')
    plt.ylabel(f'{method}_2')
    plt.savefig(f'byol_mnist_{method.lower()}.png')
    plt.close()


# %%
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets and data loaders
    train_dataset = MNISTBYOLDataset('data', train=True, download=True)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    # Initialize BYOL model
    byol_model = BYOL().to(device)
    optimizer = optim.Adam(byol_model.parameters(), lr=0.001)
    
    # Train BYOL
    print("Starting BYOL training...")
    losses = train_byol(byol_model, train_loader, optimizer, device, epochs=10)
    
    # Plot BYOL training loss
    plt.plot(losses)
    plt.title('BYOL Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('byol_training_loss.png')
    plt.close()
    
    # Initialize and train classifier
    classifier = MNISTClassifier(byol_model.online_encoder).to(device)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    print("Starting classifier fine-tuning...")
    train_classifier(classifier, train_loader, test_loader, classifier_optimizer, device, epochs=10)
    
    # Extract features and create visualizations
    features, labels = extract_features(classifier, test_loader, device)
    
    print("Generating UMAP visualization...")
    plot_embeddings(features, labels, method='UMAP')
    
    print("Generating PCA visualization...")
    plot_embeddings(features, labels, method='PCA')
    
    print("BYOL training and evaluation complete!")

if __name__ == "__main__":
    main()


