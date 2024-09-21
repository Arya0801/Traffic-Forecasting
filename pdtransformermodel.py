import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from flask import Flask, request, jsonify, render_template
from datetime import datetime
import joblib

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define functions for data loading and preprocessing
def load_pems_data(file):
    with np.load(file) as npz:
        x = npz['x']
    return x

def preprocess_pems_data(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    return data, scaler

def create_adjacency_matrix_pems(file):
    df = pd.read_csv(file)
    df['from'] = df['from'].astype(int)
    df['to'] = df['to'].astype(int)
    max_index = max(df['from'].max(), df['to'].max())
    num_nodes = max_index + 1
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for _, row in df.iterrows():
        try:
            from_idx = int(row['from'])
            to_idx = int(row['to'])
            adj_matrix[from_idx, to_idx] = row['cost']
            adj_matrix[to_idx, from_idx] = row['cost']
        except IndexError as e:
            print(f"Index error: {e} for row: {row}")
    
    return adj_matrix

# Load and preprocess PeMSD8 data
train_data = load_pems_data('train.npz')
test_data = load_pems_data('test.npz')
val_data = load_pems_data('val.npz')

# Preprocess data
train_data, scaler = preprocess_pems_data(train_data)
test_data, _ = preprocess_pems_data(test_data)
val_data, _ = preprocess_pems_data(val_data)

# Load the adjacency matrix
adj_matrix = create_adjacency_matrix_pems('adj_pemsd8.csv')
# Save the scaler
joblib.dump(scaler, 'scaler2.pkl')

# Define the Transformer model
class DynamicLongRangeTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(DynamicLongRangeTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True  # Ensure batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        print(f"Initial Input Shape: {x.size()}")  # Debug print
        batch_size, seq_length, num_nodes, input_dim = x.size()
        x = x.view(batch_size * seq_length, num_nodes, input_dim)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        transformer_out = self.transformer(x, x)
        transformer_out = transformer_out.permute(1, 0, 2)
        out = self.fc(transformer_out)
        out = out.view(batch_size, seq_length, num_nodes, -1)
        print(f"Output Shape: {out.size()}")  # Debug print
        return out

def train_model(train_data, test_data, val_data, num_nodes, input_dim, hidden_dim, num_heads, num_layers, output_dim):
    model = DynamicLongRangeTransformer(input_dim, hidden_dim, num_heads, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Convert data to tensors
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    val_data_tensor = torch.tensor(val_data, dtype=torch.float32).to(device)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_data_tensor)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    num_epochs = 5
    best_loss = float('inf')
    patience = 10
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            batch_data = batch_data[0].view(-1, 1, num_nodes, input_dim)
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # Ensure the shapes match before computing the loss
            outputs = outputs.view(-1, num_nodes, output_dim)
            batch_data = batch_data.view(-1, num_nodes, output_dim)
            
            # Print shapes before loss calculation
            print(f"Batch Data Shape: {batch_data.shape}")
            print(f"Outputs Shape: {outputs.shape}")

            loss = criterion(outputs, batch_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss}')

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    # Evaluation on the validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_data_tensor.view(-1, 1, num_nodes, input_dim))
        val_outputs = val_outputs.view(-1, num_nodes, output_dim)
        val_data_tensor = val_data_tensor.view(-1, num_nodes, output_dim)
        
        val_loss = criterion(val_outputs, val_data_tensor)
        print(f'Validation Loss: {val_loss.item()}')

        val_outputs_np = val_outputs.cpu().numpy()
        val_data_np = val_data_tensor.cpu().numpy()
        val_mae = mean_absolute_error(val_data_np, val_outputs_np)
        val_rmse = math.sqrt(mean_squared_error(val_data_np, val_outputs_np))
        print(f'Validation MAE: {val_mae}, Validation RMSE: {val_rmse}')

    # Testing
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data_tensor.view(-1, 1, num_nodes, input_dim))
        test_outputs = test_outputs.view(-1, num_nodes, output_dim)
        test_data_tensor = test_data_tensor.view(-1, num_nodes, output_dim)
        
        test_loss = criterion(test_outputs, test_data_tensor)
        print(f'Test Loss: {test_loss.item()}')

        test_outputs_np = test_outputs.cpu().numpy()
        test_data_np = test_data_tensor.cpu().numpy()
        test_mae = mean_absolute_error(test_data_np, test_outputs_np)
        test_rmse = math.sqrt(mean_squared_error(test_data_np, test_outputs_np))
        print(f'Test MAE: {test_mae}, Test RMSE: {test_rmse}')

    torch.save(model.state_dict(), 'transformer_model_pemsd8.pth')

if __name__ == '__main__':
    # Load and preprocess data
    train_data = load_pems_data('train.npz')
    test_data = load_pems_data('test.npz')
    val_data = load_pems_data('val.npz')

    # Preprocess data
    train_data, scaler = preprocess_pems_data(train_data)
    test_data, _ = preprocess_pems_data(test_data)
    val_data, _ = preprocess_pems_data(val_data)

    # Define dimensions
    num_nodes = train_data.shape[1]
    input_dim = train_data.shape[-1]
    output_dim = 2  # Define output dimension according to your problem

    # Train the model
    train_model(train_data, test_data, val_data, num_nodes, input_dim, hidden_dim=8, num_heads=8, num_layers=4, output_dim=output_dim)
