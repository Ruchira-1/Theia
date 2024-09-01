import numpy as np
from PIL import Image
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets,transforms
from sklearn.datasets import make_classification
import torch

# Generate data using make_classification()
X, y = make_classification(n_samples=1000, n_features=60*60, n_classes=10,n_clusters_per_class=1,n_informative=50*50,random_state=42)
X = X.reshape(-1,1,60,60,) 
# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # Assuming y contains class labels

# Optionally, create a PyTorch dataset
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),padding=1),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),padding=1),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),padding=1),
    
         )
        self.conv_output_size = self._calculate_conv_output_size()
        
        self.classifier =  nn.Sequential(       
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 1024),  # Assuming input size is (145, 200)
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
            nn.Sigmoid()
        )
    def _calculate_conv_output_size(self):
            # Dummy input to calculate output size
        x = torch.randn(1, 1, 60,60)  # Assuming input size is (1, 145, 200)
        x = self.features(x)
        return x.view(x.size(0), -1).size(1)
    
    def forward(self, x):
           
        x = self.features(x)
        x = self.classifier(x)
        return x
    
model = Net()

optimizer = optim.Adam(model.parameters(), lr=0.03)

criterion = nn.BCELoss()
train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

num_epochs = 10
batch_size = 32
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i in range(0, len(X_tensor), batch_size):
        inputs = X_tensor[i:i+batch_size]
        labels = y_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
         # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    # epoch_loss = running_loss / len(X_tensor)
    epoch_accuracy = 100 * correct / total
    # Print training loss per epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss:  {loss.item():.4f}, Accuracy: {epoch_accuracy:.2f}%')