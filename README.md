# Traffic Forecasting
In this project I worked on two different models for the same task i.e. traffic forecasting, to compare and know which one gives the best results so that it can be used for route planning.

Model Description

1 Graph Neural Network (GNN) Model

● Architecture: The GNN model used is the Spatiotemporal Graph Convolutional Network
(STGCN). It consists of multiple layers of spatial and temporal convolutions designed to
capture complex dependencies in traffic data.
● Training Details:
○ Loss Function: Mean Squared Error (MSE) Loss.
○ Optimizer: Adam optimizer with a learning rate of 0.0001.
● Early Stopping: Monitors validation loss to prevent overfitting.

2 Transformer-based Model

● Architecture: The Transformer-based model used is the Long Range Transformer (LRT).
It includes multi-head self-attention mechanisms and feed-forward layers to process
long-term dependencies in traffic data.
● Training Details:
○ Loss Function: Mean Squared Error (MSE) Loss.
○ Optimizer: Adam optimizer with a learning rate of 0.0001.
○ Early Stopping: Applied based on validation performance.
