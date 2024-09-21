from flask import Flask, request, jsonify, render_template
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import io
import base64
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from pdtransformermodel import preprocess_pems_data,create_adjacency_matrix_pems,DynamicLongRangeTransformer, load_pems_data

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for CUDA
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Check your CUDA installation and GPU configuration.")

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Load scaler
scaler = joblib.load('scaler2.pkl')  # Load the scaler saved during training



adj_matrix = create_adjacency_matrix_pems('adj_pemsd8.csv')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive and parse JSON input
        data = request.get_json()
        if not data:
            raise ValueError("No data received in the request")
        
        logger.info("Received data: %s", data)

        required_fields = ['source_lat', 'source_lon', 'destination_lat', 'destination_lon']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(data[field], str):
                raise ValueError(f"Field {field} must be a string")

        source_lat = float(data['source_lat'])
        source_lon = float(data['source_lon'])
        destination_lat = float(data['destination_lat'])
        destination_lon = float(data['destination_lon'])

        # Convert val_data to PyTorch tensor
        val_data_tensor = torch.from_numpy(val_data).float().to(device)

        # Flatten the val_data for easier filtering
        flattened_data = val_data_tensor.view(-1, val_data_tensor.size(-1))
        
        # Filter historical features based on latitude and longitude with tolerance
        tolerance = 1e-5
        source_lat_tensor = torch.tensor(source_lat, dtype=torch.float32).to(device)
        source_lon_tensor = torch.tensor(source_lon, dtype=torch.float32).to(device)
        
        mask = (torch.abs(flattened_data[:, 0] - source_lat_tensor) < tolerance) & \
               (torch.abs(flattened_data[:, 1] - source_lon_tensor) < tolerance)
        historical_features = flattened_data[mask]

        logger.info("Shape of historical_features: %s", historical_features.shape)
        logger.info("Historical features data: %s", historical_features)

        if historical_features.size(0) == 0:
            return jsonify({'error': 'No historical data available for the specified source location'}), 400

        # Get recent traffic data
        recent_traffic = historical_features[-1]  # Last row has the most recent data

        combined_features = recent_traffic
        
        # Adjust input_data to match the expected input dimensions
        input_data = torch.zeros((1, 1, adj_matrix.shape[0], combined_features.size(0)), dtype=torch.float32).to(device)
        input_data[0, 0, 0, :] = combined_features
        
        # Load the model
        hidden_dim = 16  # Adjusted hidden_dim based on the new model
        output_dim = 2   # Adjusted output_dim based on the new model

        model = DynamicLongRangeTransformer(input_dim=input_data.size(-1), hidden_dim=hidden_dim, num_heads=8, num_layers=4, output_dim=output_dim).to(device)
        model.load_state_dict(torch.load('transformer_model_pemsd8.pth', map_location=device))
        model.eval()
        
        # Make predictions
        with torch.no_grad():
            prediction = model(input_data)

        # Ensure prediction is a tensor
        if isinstance(prediction, torch.Tensor):
            predicted_traffic = prediction.squeeze().cpu().numpy()
        else:
            raise TypeError("Prediction must be a PyTorch tensor.")

        # Inverse normalize the predictions
        predicted_traffic = scaler.inverse_transform(predicted_traffic.reshape(-1, predicted_traffic.shape[-1])).reshape(predicted_traffic.shape)

        # Get actual traffic data from test_data for comparison
        test_data_tensor = torch.from_numpy(test_data).float().to(device)
        actual_traffic = test_data_tensor.view(-1, test_data_tensor.size(-1)).cpu().numpy()

        # Generate evaluation graph
        time_points = np.arange(len(predicted_traffic))
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, predicted_traffic, label='Predicted Traffic', color='blue', linestyle='-')
        plt.plot(time_points, actual_traffic[:len(predicted_traffic)], label='Actual Traffic', color='red', linestyle='--')
        plt.title('Traffic Prediction vs Actual Traffic')
        plt.xlabel('Time')
        plt.ylabel('Traffic')
        plt.legend()
        plt.grid(True)

        # Convert plot to PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()

        # Return JSON response with prediction and graph
        return jsonify({
            'predicted_traffic': predicted_traffic.tolist(),
            'graph': img_base64
        })

    except ValueError as ve:
        logger.error("ValueError: %s", str(ve))
        return jsonify({'error': str(ve)}), 400
    except IndexError as ie:
        logger.error("IndexError: %s", str(ie))
        return jsonify({'error': str(ie)}), 400
    except FileNotFoundError as fnfe:
        logger.error("FileNotFoundError: %s", str(fnfe))
        return jsonify({'error': 'Model file not found'}), 500
    except Exception as e:
        logger.error("Exception: %s", str(e))
        return jsonify({'error': 'An unexpected error occurred'}), 500

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

    app.run(debug=True)
