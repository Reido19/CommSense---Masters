import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import Binary_reader


#---------------------------------------------------------------------------------------
# Readin Training and Testing Data
#---------------------------------------------------------------------------------------
folder_path = "/home/torps/CommSense/Datadump/Testing/Wheelchair_Testing/6/Bin/"
complete_data, original_data = Binary_reader.get_pandas_dataFrame(folder_path)

test_path = "/home/torps/CommSense/Datadump/Testing/Wheelchair_Testing/6/Test/"
iso_test, iso_test_original = Binary_reader.get_pandas_dataFrame(test_path)


#---------------------------------------------------------------------------------------
# Seperate Targets
#---------------------------------------------------------------------------------------
Targets = original_data.pop('Target')
Targets_Test = iso_test_original.pop('Target')


#---------------------------------------------------------------------------------------
# Recombine to make Complex Datasets
#---------------------------------------------------------------------------------------
real = original_data.iloc[:, :14].values
imag = original_data.iloc[:, 14:].values

real_test = iso_test_original.iloc[:, :14].values
imag_test = iso_test_original.iloc[:, 14:].values

df_complex = real + 1j * imag
df_complex_test = real_test + 1j * imag_test


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

df_complex = df_complex[balanced_indices]
Targets = Targets[balanced_indices]


#---------------------------------------------------------------------------------------
# Randomize the data
#---------------------------------------------------------------------------------------
np.random.seed(42)
indices = np.random.permutation(len(df_complex_test))
df_complex_test = df_complex_test[indices]
Targets_Test = Targets_Test[indices]

#---------------------------------------------------------------------------------------
# Reduce Targets by a factor of 600
#---------------------------------------------------------------------------------------
reduced = Targets[::600]
reduced_array = reduced.to_numpy().reshape(-1, 1)


#---------------------------------------------------------------------------------------
# Create the Processing Function
#---------------------------------------------------------------------------------------

def preprocess_complex_df(df, label_array):
    """
    Converts df of shape (N*600, 14) complex to:
    - list of (600, 14) complex tensors
    - (N, 1) label array
    """
    images = np.split(df, len(df) // 600)
    image_tensors = [torch.tensor(img, dtype=torch.cfloat) for img in images]

    # Ensure label shape is (N,)
    label_array = np.array(label_array).reshape(-1)
    return image_tensors, label_array


#---------------------------------------------------------------------------------------
# Flatten the Dataset
#---------------------------------------------------------------------------------------

class FlattenedImageDataset(Dataset):
    def __init__(self, image_list, label_array):
        self.images = [img.reshape(-1).float() for img in image_list]  # flatten and convert to float
        self.labels = torch.tensor(label_array, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


#---------------------------------------------------------------------------------------
# Process Training and Testing Data
#---------------------------------------------------------------------------------------

train_images, train_labels = preprocess_complex_df(df_complex, reduced_array)
test_images, test_labels = preprocess_complex_df(df_complex_test, Targets_Test)

train_dataset = FlattenedImageDataset(train_images, train_labels)
test_dataset = FlattenedImageDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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
# Define MLP Model
#---------------------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2)  # 2 output classes for CrossEntropy
        )

    def forward(self, x):
        return self.model(x)


#---------------------------------------------------------------------------------------
# Instantiate the Model
#---------------------------------------------------------------------------------------
model = MLP(input_dim=600 * 14)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


#---------------------------------------------------------------------------------------
# Train the Model
#---------------------------------------------------------------------------------------
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


#---------------------------------------------------------------------------------------
# Evaluate the Model
#---------------------------------------------------------------------------------------
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        predicted = torch.argmax(outputs, dim=1)

        all_predictions.extend(predicted.tolist())
        all_labels.extend(y_batch.tolist())


#---------------------------------------------------------------------------------------
# Display some of the predictions made
#---------------------------------------------------------------------------------------
print("\nFirst 20 Predictions vs True Labels:")
for i in range(20):
    print(f"Prediction: {all_predictions[i]}, Label: {all_labels[i]}")


#---------------------------------------------------------------------------------------
# Calculate and Display the model accuracy
#---------------------------------------------------------------------------------------
correct = sum([pred == label for pred, label in zip(all_predictions, all_labels)])
total = len(all_labels)
print(f"\nTest Accuracy: {correct / total:.4f}")


