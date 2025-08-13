import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
import numpy as np
import pandas as pd

import Binary_reader

folder_path = "/home/torps/CommSense/Datadump/Testing/Wheelchair_Testing/6/Bin/"
complete_data, original_data = Binary_reader.get_pandas_dataFrame(folder_path)

test_path = "/home/torps/CommSense/Datadump/Testing/Wheelchair_Testing/6/Test/"
iso_test, iso_test_original = Binary_reader.get_pandas_dataFrame(test_path)

#---------------------------------------------------------------------------------------
# Seperate Targets
#---------------------------------------------------------------------------------------
Y_train = complete_data.pop('Target')
Y_test = iso_test.pop('Target')

X_train = complete_data
X_test = iso_test

#---------------------------------------------------------------------------------------
# Balance the labeled calsses
#---------------------------------------------------------------------------------------
idx_0 = np.where(Y_train == 0)[0]
idx_1 = np.where(Y_train == 1)[0]

min_len = min(len(idx_0), len(idx_1))
idx_0_sample = np.random.choice(idx_0, size=min_len, replace=False)
idx_1_sample = np.random.choice(idx_1, size=min_len, replace=False)


balanced_indices = np.concatenate([idx_0_sample, idx_1_sample])
np.random.shuffle(balanced_indices)


X_train = X_train.iloc[balanced_indices]
Y_train = Y_train.iloc[balanced_indices]


#---------------------------------------------------------------------------------------
# Randomize the Test data
#---------------------------------------------------------------------------------------
n = random.randint(0, 100)
np.random.seed(42)
indices = np.random.permutation(len(Y_test))
X_test = X_test.iloc[indices]
Y_test = Y_test.iloc[indices]

#---------------------------------------------------------------------------------------
# Calculate the number of each labelled class
#---------------------------------------------------------------------------------------
unique, counts = np.unique(Y_train, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Label {u}: {c} instances")

unique_test, counts_test = np.unique(Y_test, return_counts=True)

for r, b in zip(unique_test, counts_test):
    print(f"Label Test {r}: {b} instances")


X_train_real = np.real(X_train)
X_train_imag = np.imag(X_train)
X_test_real = np.real(X_test)
X_test_imag = np.imag(X_test)

X_train_combined = np.concatenate([X_train_real, X_train_imag], axis=1)
X_test_combined = np.concatenate([X_test_real, X_test_imag], axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)


# ==== Convert to PyTorch Tensors ====
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
Y_train_tensor = torch.tensor(np.array(Y_train), dtype=torch.long)
# Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
# Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),        # Binary classification → 1 output
                # nn.Sigmoid()             # Used only with BCELoss (NOT BCEWithLogitsLoss)
            )
    #     self.model = nn.Sequential(
    #         nn.Linear(input_dim, 256),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(),
    #         nn.Dropout(0.3),
    #         nn.Linear(256, 128),
    #         nn.BatchNorm1d(128),
    #         nn.ReLU(),
    #         nn.Dropout(0.3),
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 2),
    #         )


    def forward(self, x):
        return self.model(x)

model = MLP(1200)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


#---------------------------------------------------------------------------------------
# Evaluate the Model
#---------------------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = torch.argmax(predictions, dim=1)
    accuracy = accuracy_score(Y_test_tensor.numpy(), predicted_labels.numpy())
    print(f"\nTest Accuracy: {accuracy:.4f}")



#---------------------------------------------------------------------------------------
# Display some of the predictions made
#---------------------------------------------------------------------------------------
print("\nFirst 20 Predictions vs True Labels:")
for i in range(20):
     print(f"Prediction: {predicted_labels[i].item()}, True Label: {Y_test_tensor[i].item()}")
