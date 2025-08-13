import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

import Binary_reader


#---------------------------------------------------------------------------------------
# Readin Training and Testing Data
#---------------------------------------------------------------------------------------
folder_path = "/home/torps/CommSense/Datadump/Testing/Wheelchair_Testing/Final/Bin/"
complete_data, original_data = Binary_reader.get_pandas_dataFrame(folder_path)

test_path = "/home/torps/CommSense/Datadump/Testing/Wheelchair_Testing/Final/Test/"
iso_test, iso_test_original = Binary_reader.get_pandas_dataFrame(test_path)


#---------------------------------------------------------------------------------------
# Seperate Targets
#---------------------------------------------------------------------------------------
Targets = complete_data.pop('Target')
Targets_Test = iso_test.pop('Target')


#---------------------------------------------------------------------------------------
# Balance the labeled calsses
#---------------------------------------------------------------------------------------
idx_0 = np.where(Targets == 0)[0]
idx_1 = np.where(Targets == 1)[0]

min_len = min(len(idx_0), len(idx_1))
idx_0_sample = np.random.choice(idx_0, size=min_len, replace=False)
idx_1_sample = np.random.choice(idx_1, size=min_len, replace=False)


balanced_indices = np.concatenate([idx_0_sample, idx_1_sample])
np.random.shuffle(balanced_indices)


complete_data = complete_data.iloc[balanced_indices]
Targets = Targets.iloc[balanced_indices]


#---------------------------------------------------------------------------------------
# Randomize the Test data
#---------------------------------------------------------------------------------------
np.random.seed(42)
indices = np.random.permutation(len(iso_test))
iso_test = iso_test.iloc[indices]
Targets_Test = Targets_Test.iloc[indices]

#---------------------------------------------------------------------------------------
# Calculate the number of each labelled class
#---------------------------------------------------------------------------------------
unique, counts = np.unique(Targets, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Label {u}: {c} instances")

unique_test, counts_test = np.unique(Targets_Test, return_counts=True)

for r, b in zip(unique_test, counts_test):
    print(f"Label Test {r}: {b} instances")


#---------------------------------------------------------------------------------------
# Normalize using magnitude
#---------------------------------------------------------------------------------------
def normalize_complex(X):
    scaler = StandardScaler()
    mag = np.abs(X)
    mag_scaled = scaler.fit_transform(mag)
    # retain original phase
    X_unit = X / (np.abs(X) + 1e-8)  # avoid division by zero
    return mag_scaled * X_unit

X_train_norm = normalize_complex(complete_data)
X_test_norm = normalize_complex(iso_test)

print(type(X_train_norm))
print(X_train_norm.shape)


#---------------------------------------------------------------------------------------
# Calculate the number of each labelled class
#---------------------------------------------------------------------------------------
unique, counts = np.unique(Targets, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Label {u}: {c} instances")

unique_test, counts_test = np.unique(Targets_Test, return_counts=True)

for r, b in zip(unique_test, counts_test):
    print(f"Label Test {r}: {b} instances")


#---------------------------------------------------------------------------------------
# Normalize using magnitude
#---------------------------------------------------------------------------------------
def normalize_complex(X):
    scaler = StandardScaler()
    mag = np.abs(X)
    mag_scaled = scaler.fit_transform(mag)
    # retain original phase
    X_unit = X / (np.abs(X) + 1e-8)  # avoid division by zero
    return mag_scaled * X_unit

X_train_norm = normalize_complex(complete_data)
X_test_norm = normalize_complex(iso_test)

#---------------------------------------------------------------------------------------
# Convert to pyTorch Tensors
#---------------------------------------------------------------------------------------

X_train_tensor = torch.tensor(X_train_norm.values, dtype=torch.cfloat)
Y_train_tensor = torch.tensor(Targets, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_norm.values, dtype=torch.cfloat)
Y_test_tensor = torch.tensor(Targets_Test, dtype=torch.long)


#---------------------------------------------------------------------------------------
# Define Complex MLP
#---------------------------------------------------------------------------------------
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.real = nn.Linear(in_features, out_features)
        self.imag = nn.Linear(in_features, out_features)

    def forward(self, x):
        r = self.real(x.real) - self.imag(x.imag)
        i = self.real(x.imag) + self.imag(x.real)
        return torch.complex(r, i)


class ComplexMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = ComplexLinear(input_dim, 128)
        self.fc2 = ComplexLinear(128, 64)
        self.output = nn.Linear(64 * 2, 2)  # real-valued output for CrossEntropyLoss

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x.real) + 1j * torch.relu(x.imag)

        x = self.fc2(x)
        x = torch.relu(x.real) + 1j * torch.relu(x.imag)

        # Final output layer (real-valued): concatenate real and imag parts
        x_combined = torch.cat([x.real, x.imag], dim=1)  # shape: (N, 128)
        return self.output(x_combined)  # shape: (N, 2)


#---------------------------------------------------------------------------------------
# Instantiate Model
#---------------------------------------------------------------------------------------
model = ComplexMLP(600)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#---------------------------------------------------------------------------------------
# Train the Model
#---------------------------------------------------------------------------------------
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")



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
