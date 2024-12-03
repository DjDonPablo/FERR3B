import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.flattened_size = None

        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 7)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        if self.flattened_size is None:
            self.flattened_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc1 = nn.Linear(self.flattened_size, 128).to(x.device)

        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x



def load_fer2013(filepath):
    data = pd.read_csv(filepath)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = [np.asarray([int(pixel) for pixel in face.split(' ')], dtype=np.uint8).reshape(width, height) for face in pixels]
    faces = np.array(faces) / 255.0
    emotions = np.array(data['emotion'])
    return train_test_split(faces, emotions, test_size=0.2, random_state=42)


print("=> Load Dataset")
X_train, X_test, y_train, y_test = load_fer2013("fer2013.csv")

print("=> Transform Dataset")
transform = transforms.Compose([transforms.ToTensor()])
X_train = torch.tensor(X_train).unsqueeze(1).float()
X_test = torch.tensor(X_test).unsqueeze(1).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

print("=> Train model")
model = EmotionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"==> Epoch {epoch}/30, loss: {loss}")

print("=> Save Model")
torch.save(model.state_dict(), "weights.pt")
