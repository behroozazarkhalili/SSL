# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Data loading and augmentation
class MNISTBarlowTwinsDataset(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        
        return self.transform(img), self.transform(img), target

# 2. Barlow Twins model architecture
class BarlowTwinsMNIST(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
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
        self.bn = nn.BatchNorm1d(feature_dim, affine=False)

    def forward(self, x):
        z = self.encoder(x)
        z = self.bn(z)
        return z

# 3. Barlow Twins loss function
def barlow_twins_loss(z1, z2, lambda_param=5e-3):
    batch_size = z1.shape[0]
    feature_dim = z1.shape[1]
    
    c = torch.mm(z1.T, z2) / batch_size
    c_diff = (c - torch.eye(feature_dim, device=device)).pow(2)
    
    off_diagonal = c_diff.flatten()[:-1].view(feature_dim - 1, feature_dim + 1)[:, 1:].flatten()
    loss = c_diff.diagonal().sum() + lambda_param * off_diagonal.sum()
    
    return loss

# 4. Pre-training function
def pretrain_barlow_twins(model, train_loader, optimizer, epochs=10):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for data1, data2, _ in progress_bar:
            data1, data2 = data1.to(device), data2.to(device)
            
            optimizer.zero_grad()
            
            z1 = model(data1)
            z2 = model(data2)
            
            loss = barlow_twins_loss(z1, z2)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    
    return losses

# 5. Classifier for fine-tuning
class MNISTClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = pretrained_model.encoder
        self.classifier = nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

# 6. Fine-tuning function
def finetune(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}, Test Accuracy: {accuracy:.2f}%')
        model.train()

# 7. Feature extraction function
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Extracting features'):
            data = data.to(device)
            feature = model(data)
            features.append(feature.cpu().numpy())
            labels.append(target.numpy())
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
    plt.savefig(f'barlow_twins_mnist_{method.lower()}.png')
    plt.close()

# 9. Main execution
def main():
    # Hyperparameters
    batch_size = 256
    feature_dim = 128
    pretrain_epochs = 50
    finetune_epochs = 10
    
    # Load MNIST dataset for pre-training
    train_dataset = MNISTBarlowTwinsDataset('data', train=True, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize model and optimizer for pre-training
    model = BarlowTwinsMNIST(feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Pre-train the model
    print("Starting pre-training...")
    pretrain_losses = pretrain_barlow_twins(model, train_loader, optimizer, epochs=pretrain_epochs)

    # Plot pre-training loss
    plt.plot(pretrain_losses)
    plt.title('Barlow Twins Pre-training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('barlow_twins_pretrain_loss.png')
    plt.close()

    # Save the pre-trained model
    torch.save(model.state_dict(), 'barlow_twins_pretrained_mnist.pth')
    print("Pre-training complete!")

    # Prepare data for fine-tuning
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize classifier for fine-tuning
    classifier = MNISTClassifier(model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # Fine-tune the model
    print("Starting fine-tuning...")
    finetune(classifier, train_loader, test_loader, criterion, optimizer, epochs=finetune_epochs)

    print("Fine-tuning complete!")

    # Extract features for visualization
    features, labels = extract_features(classifier.encoder, test_loader)

    # Plot UMAP
    print("Generating UMAP visualization...")
    plot_embeddings(features, labels, method='UMAP')

    # Plot PCA
    print("Generating PCA visualization...")
    plot_embeddings(features, labels, method='PCA')

    print("Visualizations complete!")

if __name__ == "__main__":
    main()


