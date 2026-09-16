import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cpu")

INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

class DigitNN(nn.Module):
    def __init__(self):
        super(DigitNN,self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


#  DATASET LOADER
transform = transforms.Compose([
    transforms.ToTensor() #  Convert to [0.0, 1.0]
])

train_dataset = datasets.MNIST("./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST("./data", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model and Adam's Optimizer
model = DigitNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Start Training  ")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on Test Set: {100 * correct / total:.2f}%")


w1 = model.fc1.weight.detach().cpu().numpy()
b1 = model.fc1.bias.detach().cpu().detach().numpy()
w2 = model.fc2.weight.detach().cpu().numpy()
b2 = model.fc2.bias.detach().cpu().detach().numpy()

w1_flat = w1.flatten()
w2_flat = w2.flatten()

with open("weights.bin", "wb") as f:
    f.write(w1_flat.astype('float32').tobytes())
    f.write(b1.astype('float32').tobytes())

    f.write(w2_flat.astype('float32').tobytes())
    f.write(b2.astype('float32').tobytes())

print("Saved successfully!")