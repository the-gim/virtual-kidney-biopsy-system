import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
import numpy as np
import time

torch.set_num_threads(1)

class CNN1D(nn.Module):
    def __init__(self, input_dim=20, num_classes=2):
        super(CNN1D, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        conv_out_size = ((input_dim - 3 + 1) // 2)
        self.fc1 = nn.Linear(32 * conv_out_size, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("Generating dummy data...")
X = np.random.rand(16310, 20).astype(np.float32)
y = np.random.randint(0, 2, size=(16310,)).astype(np.int64)

print("Training CNN on CPU with threads=1...")
cnn_model = NeuralNetClassifier(
    module=CNN1D,
    module__input_dim=20,
    module__num_classes=2,
    max_epochs=20,
    lr=0.001,
    batch_size=256,
    optimizer=torch.optim.Adam,
    criterion=nn.CrossEntropyLoss,
    iterator_train__shuffle=True,
    device='cpu',
    verbose=1
)

start = time.time()
cnn_model.fit(X, y)
print(f"Finished in {time.time() - start:.2f} seconds.")
