from flask import Flask, request, jsonify, render_template
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from gnnmodel import pems_data, load_pems_data, adj_matrix, GNNRoutePlanning  # Adjust import paths as necessary
import logging
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        required_fields = ['source_lat', 'source_lon', 'destination_lat', 'destination_lon', 'time_of_travel']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(data[field], str):
                raise ValueError(f"Field {field} must be a string")

        source_lat = float(data['source_lat'])
        source_lon = float(data['source_lon'])
        destination_lat = float(data['destination_lat'])
        destination_lon = float(data['destination_lon'])
        
        logger.info("Received time_of_travel: %s", data['time_of_travel'])

        # Attempt to parse the time_of_travel in ISO 8601 format
        time_of_travel = datetime.fromisoformat(data['time_of_travel'])
        logger.info("Parsed time_of_travel: %s", time_of_travel)

        # Debugging information
        logger.info("Shape of pems_data: %s", pems_data.shape)

        # Print some of the coordinates in pems_data for debugging
        logger.info("Sample latitude coordinates in pems_data: %s", pems_data[:, :, :, 0].flatten()[:10])
        logger.info("Sample longitude coordinates in pems_data: %s", pems_data[:, :, :, 1].flatten()[:10])

        # Tolerance for matching coordinates
        tolerance = 1e-5

        # Flatten the pems_data for easier filtering
        flattened_data = pems_data.reshape(-1, 2)
        
        # Filter historical features based on latitude and longitude with tolerance
        mask = (np.abs(flattened_data[:, 0] - source_lat) < tolerance) & (np.abs(flattened_data[:, 1] - source_lon) < tolerance)
        historical_features = flattened_data[mask]

        logger.info("Shape of historical_features: %s", historical_features.shape)
        logger.info("Historical features data: %s", historical_features)

        # Handle case with no historical data
        if historical_features.size == 0:
            return jsonify({'error': 'No historical data available for the specified source location'}), 400

        # Get recent traffic data
        recent_traffic = historical_features[-1]  # Last row has the most recent data

        # Prepare time features
        time_features = pd.DataFrame([{
            'hour': time_of_travel.hour,
            'day': time_of_travel.day,
            'month': time_of_travel.month,
            'year': time_of_travel.year,
            'weekday': time_of_travel.weekday()
        }])
        
        # Standardize time features
        scaler = StandardScaler()
        time_features = scaler.fit_transform(time_features)
        
        # Ensure time_features shape matches expected input dimension
        time_features = time_features.flatten()[:0]  # Take only 5 features to match the input dimension

        # Combined features for the model
        combined_features = np.concatenate((recent_traffic, time_features))

        # Ensure input_data shape matches the model's expected input
        num_nodes = adj_matrix.shape[0]
        input_data = np.zeros((1, 1, num_nodes, combined_features.shape[0]))
        input_data[0, 0, 0, :] = combined_features
        
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        # Load the model
        hidden_dim = 8  # Assuming the hidden_dim used during training
        output_dim = 2  # Assuming the output_dim used during training

        # Ensure model is loaded on CPU
        device = torch.device('cpu')
        input_dim = combined_features.shape[0]  # Set input_dim to match the combined features length
        model = GNNRoutePlanning(num_nodes, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        # Print model summary before loading state dict
        print(model)

        # Load the model state dict with map_location
        state_dict = torch.load('gnn_model_pemsd8.pth', map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Make predictions
        with torch.no_grad():
            prediction = model(input_tensor.to(device), torch.tensor(adj_matrix, dtype=torch.float32).to(device))

        # Extract and return the prediction
        predicted_traffic = prediction.squeeze().cpu().numpy()
        return jsonify({'predicted_traffic': predicted_traffic.tolist()})

    except Exception as e:
        logger.error("Error: %s", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
