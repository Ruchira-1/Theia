import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.datasets import make_classification
import sys
sys.path.append('/Users/ruchira/Desktop/Torch Programs')
from theia import Theia_callback
# Generate synthetic image data
num_samples = 1000
image_height = 145
image_width = 200
num_channels = 1  # Grayscale image

X, y = make_classification(n_samples=num_samples, n_features=image_height * image_width * num_channels,
                           n_classes=2, n_clusters_per_class=1, random_state=42)

# Reshape the data into image format
X_images = X.reshape(-1, num_channels, image_height, image_width)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_images, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Define a simple CNN model
class Hl_Model(nn.Module):
    
    # torch.cuda.set_device(0)

    def __init__(self):
        super().__init__()
       
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
             nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )   
        self.classifier =  nn.Sequential(   
            nn.Flatten(),
            nn.Linear(3840, 128),
            # nn.BatchNorm2d(128),  # Assuming input size is (145, 200)
            nn.ReLU(),
            nn.Linear(128, 3),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
       
        x = self.features(x)
        x = self.classifier(x)
        return x

# Create an instance of the model
model = Hl_Model()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Theia_callback.check(X_images, X_images,model, criterion, optimizer,32,1,0)
# Training loop
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