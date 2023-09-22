import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim

#Load the dataset
df = pd.read_csv("og_ucf101_features.csv")
df['video_features'] = df['video_features'].apply(lambda x: list(map(float, x.strip("[]").split())))
df['video_features'] = df['video_features'].apply(np.array)

# Convert them into numpy arrays
X = np.stack(df['video_features'].values)

le = LabelEncoder()
df['video_label'] = le.fit_transform(df['video_label'])

# Split the data
X = np.stack(df['video_features'].values)
y = df['video_label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()

        self.layer1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.layer4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.output_layer = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = nn.ReLU()(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = nn.ReLU()(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = nn.ReLU()(self.bn4(self.layer4(x)))
        x = self.output_layer(x)
        return x

# Initialize the model
input_dim = X_train.shape[1]
output_dim = len(set(y_train.tolist()))
model = NeuralNet(input_dim, output_dim)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Validate model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    total += y_test.size(0)
    correct += (predicted == y_test).sum().item()

    print(f'Test Accuracy: {(100 * correct / total):.2f}%')