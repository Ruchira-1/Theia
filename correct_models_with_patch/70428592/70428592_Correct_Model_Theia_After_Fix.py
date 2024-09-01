import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
#from keras.datasets import mnist
from torchvision import datasets
import os
import pandas as pd 
import sys
sys.path.append('/Users/ruchira/Desktop/Torch Programs')
from theia import Theia_callback
from sklearn.datasets import make_classification
#import matplotlib.pyplot as plt 

# Generate data using make_classification()
X, y = make_classification(n_samples=1000, n_features=50*50*1, n_classes=2,n_clusters_per_class=1,n_informative=40*40*1,random_state=42)
X = X.reshape(-1,1,50,50,) 
# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # Assuming y contains class labels

# Optionally, create a PyTorch dataset
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) , 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5,stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
    
         )
        
        self.conv_output_size = self._calculate_conv_output_size()
        
        self.classifier =  nn.Sequential(       
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 500),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.BatchNorm1d(500),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout1d(0.2),
            nn.Linear(500, 2),
            nn.Sigmoid()
           
        )

    def _calculate_conv_output_size(self):
            # Dummy input to calculate output size
        x = torch.randn(1, 1, 50,50)  # Assuming input size is (1, 145, 200)
        x = self.features(x)
        return x.view(x.size(0), -1).size(1)
    
    def forward(self, x):
           
        x = self.features(x)
        x = self.classifier(x)
        return x

model = Net()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# Theia_callback.check(X, X,model, criterion, optimizer,32,1,0)

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