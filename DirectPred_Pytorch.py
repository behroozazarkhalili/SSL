import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

class Config:
    def __init__(self):
        self.batch_size = 256
        self.feature_dim = 256  # Increased from 128
        self.pred_dim = 256  # Increased from 128
        self.learning_rate = 0.05  # Increased from 0.03
        self.momentum = 0.99
        self.weight_decay = 1e-4
        self.epochs = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNISTDataset(Dataset):
    def __init__(self, split='train'):
        self.dataset = datasets.MNIST(root='./data', train=(split == 'train'), download=True,
                                      transform=transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return {'image': img, 'label': label}

    def augment(self, image):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # Added vertical flip
            transforms.RandomRotation(15),    # Added rotation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Added translation
        ])
        return transform(image)

class Encoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),  # Increased filters
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim),
            nn.BatchNorm1d(feature_dim)  # Added normalization to the output
        )

    def forward(self, x):
        return self.encoder(x)

class Predictor(nn.Module):
    def __init__(self, feature_dim, pred_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, feature_dim)
        )

    def forward(self, x):
        return self.predictor(x)

class DirectPred(nn.Module):
    def __init__(self, feature_dim, pred_dim):
        super().__init__()
        self.encoder = Encoder(feature_dim)
        self.predictor = Predictor(feature_dim, pred_dim)
        self.target_encoder = Encoder(feature_dim)

        # Initialize target_encoder with encoder's parameters
        self.target_encoder.load_state_dict(self.encoder.state_dict())

        # Freeze target_encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        with torch.no_grad():
            t1 = self.target_encoder(x1)
            t2 = self.target_encoder(x2)
        return p1, p2, t1.detach(), t2.detach()

class DirectPredTrainer:
    def __init__(self, model: DirectPred, train_loader: DataLoader, optimizer: optim.Optimizer, device: torch.device, momentum: float):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.momentum = momentum

    def train(self, epochs: int):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                x = batch['image'].to(self.device)
                x1 = self.train_loader.dataset.augment(x)
                x2 = self.train_loader.dataset.augment(x)
                
                self.optimizer.zero_grad()
                p1, p2, t1, t2 = self.model(x1, x2)
                loss = self._directpred_loss(p1, p2, t1, t2)
                loss.backward()
                self.optimizer.step()
                
                self._update_target_encoder()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(self.train_loader):.4f}")

    def _directpred_loss(self, p1, p2, t1, t2):
        loss = nn.functional.mse_loss(p1, t2) + nn.functional.mse_loss(p2, t1)
        return loss

    def _update_target_encoder(self):
        for online_params, target_params in zip(self.model.encoder.parameters(), self.model.target_encoder.parameters()):
            target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data

class DirectPredEvaluator:
    def __init__(self, model: DirectPred, train_loader: DataLoader, test_loader: DataLoader, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        linear_metrics = self._linear_evaluation()
        knn_metrics = self._knn_evaluation()
        return linear_metrics, knn_metrics

    def _linear_evaluation(self):
        train_features, train_labels = self._extract_features(self.train_loader)
        test_features, test_labels = self._extract_features(self.test_loader)

        classifier = nn.Linear(train_features.shape[1], 10).to(self.device)  # 10 classes for MNIST
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(100):  # Train for 100 epochs
            for i in range(0, len(train_features), 256):
                batch_features = train_features[i:i+256].to(self.device)
                batch_labels = train_labels[i:i+256].to(self.device)
                optimizer.zero_grad()
                outputs = classifier(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

        classifier.eval()
        with torch.no_grad():
            test_outputs = classifier(test_features.to(self.device))
            _, predicted = torch.max(test_outputs, 1)
            accuracy = accuracy_score(test_labels.cpu().numpy(), predicted.cpu().numpy())
        return {'accuracy': accuracy}

    def _knn_evaluation(self, k=5):
        train_features, train_labels = self._extract_features(self.train_loader)
        test_features, test_labels = self._extract_features(self.test_loader)

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_features.cpu().numpy(), train_labels.cpu().numpy())
        predictions = knn.predict(test_features.cpu().numpy())
        accuracy = accuracy_score(test_labels.cpu().numpy(), predictions)
        return {'accuracy': accuracy}

    def _extract_features(self, loader):
        self.model.eval()
        features, labels = [], []
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device)
                feature = self.model.encoder(images)
                features.append(feature.cpu())
                labels.append(batch['label'])
        return torch.cat(features), torch.cat(labels)

def main():
    config = Config()

    train_dataset = MNISTDataset(split='train')
    test_dataset = MNISTDataset(split='test')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)

    model = DirectPred(config.feature_dim, config.pred_dim).to(config.device)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    
    # Added learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    trainer = DirectPredTrainer(model, train_loader, optimizer, config.device, config.momentum)
    evaluator = DirectPredEvaluator(model, train_loader, test_loader, config.device)

    print("Starting DirectPred training...")
    for epoch in range(config.epochs):
        trainer.train(1)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"\nEvaluating the model at epoch {epoch + 1}...")
            linear_metrics, knn_metrics = evaluator.evaluate()
            print("\nLinear Evaluation Metrics:")
            for key, value in linear_metrics.items():
                print(f"{key.capitalize()}: {value:.4f}")
            print("\nKNN Evaluation Metrics:")
            for key, value in knn_metrics.items():
                print(f"{key.capitalize()}: {value:.4f}")

    print("\nFinal Evaluation:")
    linear_metrics, knn_metrics = evaluator.evaluate()
    print("\nLinear Evaluation Metrics:")
    for key, value in linear_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")
    print("\nKNN Evaluation Metrics:")
    for key, value in knn_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()
