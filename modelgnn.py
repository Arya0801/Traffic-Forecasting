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
import logging
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


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
joblib.dump(scaler, 'scaler.pkl')

# Define model classes
class GraphSAGE(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphSAGE, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.fc(x)
        return x

class GNNRoutePlanning(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim):
        super(GNNRoutePlanning, self).__init__()
        self.sage1 = GraphSAGE(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(num_nodes)
        self.rnn = nn.LSTM(hidden_dim * num_nodes, hidden_dim * num_nodes, batch_first=True)
        self.sage2 = GraphSAGE(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x, adj):
        batch_size, seq_length, num_nodes, input_dim = x.size()
        x = x.view(batch_size * seq_length, num_nodes, input_dim)
        h = torch.relu(self.bn1(self.sage1(x, adj)))
        h = h.view(batch_size, seq_length, -1)  # Reshape to (batch_size, seq_length, num_nodes * hidden_dim)
        h, _ = self.rnn(h)
        h = h.view(batch_size * seq_length, num_nodes, -1)  # Reshape back to (batch_size * seq_length, num_nodes, hidden_dim)
        out = self.sage2(h, adj)
        out = out.view(batch_size, seq_length, num_nodes, self.output_dim)
        return out

def train_model(train_data, test_data, val_data, adj_matrix, num_nodes, input_dim, hidden_dim, output_dim):
    model = GNNRoutePlanning(num_nodes, input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Convert data to tensors
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    val_data_tensor = torch.tensor(val_data, dtype=torch.float32).to(device)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_data_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    num_epochs = 25
    best_loss = float('inf')
    patience = 10
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            batch_data = batch_data[0].view(-1, 1, num_nodes, input_dim)
            optimizer.zero_grad()
            outputs = model(batch_data, adj_matrix)
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
        val_outputs = model(val_data_tensor.view(-1, 1, num_nodes, input_dim), adj_matrix)
        val_loss = criterion(val_outputs, val_data_tensor.view(-1, 1, num_nodes, input_dim))
        print(f'Validation Loss: {val_loss.item()}')

        val_outputs_np = val_outputs.cpu().numpy()
        val_data_np = val_data_tensor.cpu().numpy()
        val_outputs_np = val_outputs_np.reshape(-1, val_outputs_np.shape[-1])
        val_data_np = val_data_np.reshape(-1, val_data_np.shape[-1])

        val_mae = mean_absolute_error(val_data_np, val_outputs_np)
        val_rmse = math.sqrt(mean_squared_error(val_data_np, val_outputs_np))
        print(f'Validation MAE: {val_mae}, Validation RMSE: {val_rmse}')

    # Testing
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data_tensor.view(-1, 1, num_nodes, input_dim), adj_matrix)
        test_loss = criterion(test_outputs, test_data_tensor.view(-1, 1, num_nodes, input_dim))
        print(f'Test Loss: {test_loss.item()}')

        test_outputs_np = test_outputs.cpu().numpy()
        test_data_np = test_data_tensor.cpu().numpy()
        test_outputs_np = test_outputs_np.reshape(-1, test_outputs_np.shape[-1])
        test_data_np = test_data_np.reshape(-1, test_data_np.shape[-1])

        test_mae = mean_absolute_error(test_data_np, test_outputs_np)
        test_rmse = math.sqrt(mean_squared_error(test_data_np, test_outputs_np))
        print(f'Test MAE: {test_mae}, Test RMSE: {test_rmse}')

    torch.save(model.state_dict(), 'gnn_model_pemsd8_1.pth')


if __name__ == '__main__':
    # Load and preprocess data
    train_data = load_pems_data('train.npz')
    test_data = load_pems_data('test.npz')
    val_data = load_pems_data('val.npz')

    # Preprocess data
    train_data, scaler = preprocess_pems_data(train_data)
    test_data, _ = preprocess_pems_data(test_data)
    val_data, _ = preprocess_pems_data(val_data)

    # Load the adjacency matrix
    adj_matrix = create_adjacency_matrix_pems('adj_pemsd8.csv')

    # Move adjacency matrix to the appropriate device
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)

    # Define dimensions
    num_nodes = adj_matrix.shape[0]
    input_dim = train_data.shape[-1]

    # Train the model
    train_model(train_data, test_data, val_data, adj_matrix, num_nodes, input_dim, hidden_dim=8, output_dim=2)





