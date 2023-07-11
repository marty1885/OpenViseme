import torch
import numpy as np
import math
from sklearn.model_selection import train_test_split

mel = np.load('DataSet/mel.npy')
label = np.load('DataSet/label.npy')

def onehot(label):
    onehot = np.zeros((len(label), 15))
    for i in range(len(label)):
        onehot[i][label[i]] = 1
    return onehot
label = onehot(label)

class Model(torch.nn.Module):
    def __init__(self, n_features):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(n_features, 32)
        self.l2 = torch.nn.Linear(32, 32)
        self.l3 = torch.nn.Linear(16, 15)
        self.norm1 = torch.nn.BatchNorm1d(32)
        self.norm2 = torch.nn.BatchNorm1d(32)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.005)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.norm1(x)
        x = self.relu(self.l2(x))
        x = self.norm2(x)
        x = self.softmax(self.l3(x))
        return x

n_features = 24
model = Model(n_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 512

x_train, x_test, y_train, y_test = train_test_split(mel, label, test_size=0.2, random_state=42)
x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

y_test_label = torch.argmax(y_test, dim=1)

lowest_loss = 1000000
best_model = None

for epoch in range(600):
    epoch_loss = 0
    for i in range(0, len(x_train), batch_size):
        x = x_train[i:i+batch_size]
        y = y_train[i:i+batch_size]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / math.ceil(len(mel) / batch_size)

    y_pred = model(x_test)
    loss = loss_fn(y_pred, y_test)
    y_pred = torch.argmax(y_pred, dim=1)
    pred = (y_pred == y_test_label).sum().item()
    print(f'\33[2K\rEpoch: {epoch+1}, Loss: {epoch_loss:.4f}, Test Accuracy: {pred/len(y_test):.4f}, Test Loss: {loss.item():.4f}', end='')

    if loss.item() < lowest_loss:
        lowest_loss = loss.item()
        best_model = model.state_dict()

print(f"\nLowest Loss: {lowest_loss:.4f}")

model = Model(n_features)
model.load_state_dict(best_model)
y_pred = model(x_test)
y_pred = torch.argmax(y_pred, dim=1)
pred = (y_pred == y_test_label).sum().item()
print(f"Best test Accuracy: {pred/len(y_test):.4f}")
torch.save(model.state_dict(), 'model.pth')
