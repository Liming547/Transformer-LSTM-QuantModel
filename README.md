#  =========== LSTM-Transformer Model for Market Data Prediction =========== #
# It models multivariate time-series market data spanning several years.

# Highlights
- **Scalable preprocessing with Dask:** mitigates out-of-memory issues by parallelizing feature engineering and enabling chunked.
- **Hybrid deep model (Transformer + LSTM):** captures both long-range dependencies and local temporal dynamics in noisy market signals.
- **Trend Normalization (Chen et al.):** reproduces the paper’s normalization to suppress non-stationary trends and fluctuations, improving downstream performance.

# Notes: 
- Dataset-specific feature engineering has been removed.  
  This repo focuses on the model and the training process.


# Reference: https://dl.acm.org/doi/abs/10.1145/3534678.3539234?casa_token=rpp2ybvw7McAAAAA:deLdYZ_INhc8xbwsk-aBG3NqakTgxp6Z7Zp-J18Na-_XTgAqdeyo_G1aQv8UvyWi9B8GVBifn1pmSA


# Repo Content
# ---------------------------------------------

| File          | Description |
|------------------------|-------------|
| `dlmodel.py`           | Define LSTM-Transformer model architectures. |
| `train.py`             | Train and save checkpoints. |
| `utils.py`             | Define Utility functions for data loading, preprocessing, etc. |
| `config.yaml`          | Define paths and model/training hyperparameters. |

# Configuration file
# ---------------------------------------------

  - `train_data_path` (string type): the folder containing training data.

  - `save_model_path` (string type): path to save the checkpoint.

  - `lr` : learning rate

  - `epoch_num`: number of training epoches

  - `device`: the device used, cpu, cuda, or mps

  - `lookback`: sliding window size

  - `batch_size`: traing batch size

  - `model_dim`: transformer layer dimension

  - `nhead`: number of attention head

  - `nlayer`: number of attention layer 

  - `dropout`: drop out rate 
  
  - `used_features`: Features used in the models
