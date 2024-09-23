# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class IMDBDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("imdb", split=split)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = 256

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.tokenizer(item['text'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

    def augment(self, input_ids, attention_mask):
        # Random masking
        mask_prob = 0.15
        mask_token_id = self.tokenizer.mask_token_id
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < mask_prob) * (input_ids != self.tokenizer.pad_token_id)
        input_ids_aug = input_ids.clone()
        input_ids_aug[mask_arr] = mask_token_id

        return {
            'input_ids': input_ids_aug,
            'attention_mask': attention_mask.clone()
        }


class ComplexNLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Initialize DistilBERT configuration
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
        # Create DistilBERT model with random weights
        self.bert = DistilBertModel(config)
        # Initialize weights randomly
        self.bert.init_weights()
        self.fc = nn.Linear(config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token representation
        return self.fc(pooled_output)

class SimSiamNLP(nn.Module):
    def __init__(self, feature_dim, pred_dim):
        super().__init__()
        self.encoder = ComplexNLPEncoder(feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.prediction = nn.Sequential(
            nn.Linear(feature_dim, pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, feature_dim)
        )

    def forward(self, x1, x2):
        z1 = self.projection(self.encoder(input_ids=x1['input_ids'], attention_mask=x1['attention_mask']))
        z2 = self.projection(self.encoder(input_ids=x2['input_ids'], attention_mask=x2['attention_mask']))
        p1 = self.prediction(z1)
        p2 = self.prediction(z2)
        return p1, p2, z1.detach(), z2.detach()

def simsiam_loss(p1, p2, z1, z2):
    return -0.5 * (nn.functional.cosine_similarity(p1, z2.detach(), dim=-1).mean() +
                   nn.functional.cosine_similarity(p2, z1.detach(), dim=-1).mean())

def train_simsiam(model, train_loader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x1 = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            x2 = train_loader.dataset.augment(batch['input_ids'], batch['attention_mask'])
            x2 = {k: v.to(device) for k, v in x2.items()}
            
            optimizer.zero_grad()
            p1, p2, z1, z2 = model(x1, x2)
            loss = simsiam_loss(p1, p2, z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

def pretext_task_loss(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            x1 = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            x2 = dataloader.dataset.augment(batch['input_ids'], batch['attention_mask'])
            x2 = {k: v.to(device) for k, v in x2.items()}
            p1, p2, z1, z2 = model(x1, x2)
            loss = simsiam_loss(p1, p2, z1, z2)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def linear_evaluation_accuracy(encoder, train_loader, test_loader, device):
    encoder.eval()
    
    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                feature = encoder(input_ids, attention_mask)
                features.append(feature.cpu())
                labels.append(batch['label'])
        return torch.cat(features), torch.cat(labels)

    train_features, train_labels = extract_features(train_loader)
    test_features, test_labels = extract_features(test_loader)

    classifier = nn.Linear(train_features.shape[1], 2).to(device)  # Binary classification
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train classifier
    for epoch in range(10):
        for i in range(0, len(train_features), 64):
            batch_features = train_features[i:i+64].to(device)
            batch_labels = train_labels[i:i+64].to(device)
            optimizer.zero_grad()
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        test_outputs = classifier(test_features.to(device))
        _, predicted = torch.max(test_outputs, 1)
        accuracy = accuracy_score(test_labels.cpu(), predicted.cpu())
    return accuracy

def knn_evaluation_accuracy(encoder, train_loader, test_loader, device, k=5):
    encoder.eval()
    
    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                feature = encoder(input_ids, attention_mask)
                features.append(feature.cpu().numpy())
                labels.append(batch['label'].numpy())
        return np.vstack(features), np.concatenate(labels)

    train_features, train_labels = extract_features(train_loader)
    test_features, test_labels = extract_features(test_loader)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, train_labels)
    predictions = knn.predict(test_features)
    return accuracy_score(test_labels, predictions)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Loading
    train_dataset = IMDBDataset(split='train')
    test_dataset = IMDBDataset(split='test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model Initialization
    feature_dim = 256
    pred_dim = 128
    model = SimSiamNLP(feature_dim, pred_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training
    print("Starting SimSiam training...")
    train_simsiam(model, train_loader, optimizer, device, epochs=5)

    # Evaluation
    print("\nEvaluating the model...")
    pretext_loss = pretext_task_loss(model, test_loader, device)
    linear_acc = linear_evaluation_accuracy(model.encoder, train_loader, test_loader, device)
    knn_acc = knn_evaluation_accuracy(model.encoder, train_loader, test_loader, device)

    print(f"Pretext Task Loss: {pretext_loss:.4f}")
    print(f"Linear Evaluation Accuracy: {linear_acc:.4f}")
    print(f"KNN Evaluation Accuracy: {knn_acc:.4f}")

if __name__ == "__main__":
    main()


