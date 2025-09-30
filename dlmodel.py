import torch
from torch import nn
from typing import Dict

class Detrend(nn.Module):
    """
    Detrend a time series by subtracting a moving average (per feature).
    Reproduce algo in this paper: https://dl.acm.org/doi/abs/10.1145/3534678.3539234?casa_token=rpp2ybvw7McAAAAA:deLdYZ_INhc8xbwsk-aBG3NqakTgxp6Z7Zp-J18Na-_XTgAqdeyo_G1aQv8UvyWi9B8GVBifn1pmSA
    
    Parameters:
        window: moving-average window size.
        step: stride for the moving average (usually 1).

    Input:
        x: (B, L, E) or (L, E). If (L, E) is given, it's treated as a single batch.

    Returns:
        detrended: same shape as input
        trend: same shape as input
    """
        

    def __init__(self, window=5, step=1):
        super().__init__()
        self.window = window
        self.step = step
        self.pool = nn.AvgPool1d(kernel_size=window, stride=step, padding=0)

    def forward(self, x):
        x = x.float()
        squeezed = False

        if x.ndim == 2:
            squeezed = True
            x = x.unsqueeze(0)
        elif x.ndim != 3:  # Raise an error for invalid input dimensions
            raise ValueError(f"Input must be 2D or 3D, but got {x.shape}")

        x_t = x.permute(0, 2, 1)  
        pad_size = (self.window - 1) // 2  
        front = x_t[:, :, 0:1].repeat(1, 1, pad_size)  
        end   = x_t[:, :, -1:].repeat(1, 1, pad_size)  
        x_pad = torch.cat([front, x_t, end], dim=2)


        trend = self.pool(x_pad)
        trend = trend.permute(0, 2, 1)
        x_detrend = x - trend

        if squeezed :
            x_detrend = x_detrend.squeeze(0)
            trend = trend.squeeze(0)

        return x_detrend, trend


class Trend_Norm(nn.Module):
    """
    Trend-aware normalization:
      1) Detrend with moving average
      2) Normalize by per-sequence std
      3) Add learnable polynomial trend T(position) with per-feature betas
    Reproduce algo in this paper: https://dl.acm.org/doi/abs/10.1145/3534678.3539234?casa_token=rpp2ybvw7McAAAAA:deLdYZ_INhc8xbwsk-aBG3NqakTgxp6Z7Zp-J18Na-_XTgAqdeyo_G1aQv8UvyWi9B8GVBifn1pmSA
    
    Parameters:
        dimension: feature size E
        order:     polynomial order (>= 0)
        window:    detrend window
        eps:       numerical stability

    Input:
        tensor: (B, L, E) or (L, E). For clarity, prefer (B, L, E).
    """

    def __init__(self, dimension, order=1, window=5, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.order = order
        self.gamma = nn.Parameter(torch.ones(dimension))
        # Learnable coefficients for polynomial trend (beta)
        # beta_0, beta_1, ... beta_order (each shape: (E,))
        self.betas = nn.ParameterList([nn.Parameter(torch.zeros(dimension)) for _ in range(order + 1)])
        self.detrend = Detrend(window=window)


    def forward(self, tensor: torch.Tensor) -> torch.Tensor:

        x = tensor.float()
        squeezed = False

        if x.ndim == 2:
            x = x.unsqueeze(0)  # Shape: (1, L, E)
            squeezed = True
        elif x.ndim != 3:
            raise ValueError(f"Input must be 2D or 3D, but got {tensor.shape}")

        B, L, E = x.shape

        detrended, trend = self.detrend(x)
        std = detrended.std(dim=1, unbiased=False, keepdim=True) # (B, 1, E)

        position = torch.arange(0.0, L, 1.0, device=x.device) / L
        T = torch.zeros((B, L, E), device=x.device)

        for i in range(self.order + 1):
            beta_i = self.betas[i]
            trend_i = (position ** i).unsqueeze(1) * beta_i.unsqueeze(0)
            T += trend_i.unsqueeze(0).repeat(B, 1, 1)

        x_normalized = self.gamma * detrended / (std + self.eps) + T

        if squeezed:
            x_normalized = x_normalized.squeeze(0)

        return x_normalized


class FeatureExtractor(nn.Module):    
    """
    Per-feature encoder:
      - project scalar series (B, L) -> (B, L, D) -> (B, L, 2D)
      - transformer encoder with residual
      - LSTM summarization over time -> (B, 2D)

    Architecture:
      → Linear(1→D) → ReLU → Trend_Norm(D)
      → Linear(D→2D) → Trend_Norm(2D)
      → TransformerEncoder(num_layers=nlayer, nhead)
      → Dropout → Residual add
      → Trend_Norm(2D) → ReLU
      → LSTM(input=2D, hidden=2D) → Trend_Norm(2D)

    Notes:
      - B = Batch size
      - D = config['model_dim'], hidden dimension
      - L = config['lookback'], sliding window size

    """

    def __init__(self, config):
        super().__init__()
        D = config['model_dim']
        L = int(config['lookback'])
        self.embedding = nn.Sequential(
            nn.Linear(1, D),
            nn.ReLU(),
            Trend_Norm(D, L),
            nn.Linear(D, 2 * D),
            Trend_Norm(2 * D, L),
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=2 * D,
                                                    nhead=config['nhead'],
                                                    batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['nlayer'])
        self.dropout = nn.Dropout(config['dropout'])

        self.TrendNorm = Trend_Norm(2 * D, L)
        self.activation = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(input_size=2 * D, hidden_size=2 * D, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: (B, L)  single scalar feature over time

        Returns:
            (B, 2D) vector representation
        """

        x_proj = self.embedding(x.unsqueeze(-1))        # (B, L, 2D)
        x_encoded  = self.encoder(x_proj)
        x_dropout  = self.dropout(x_encoded)
        x_dropout  = x_dropout + x_proj  
        
        # LSTM
        x_lstm = self.TrendNorm(x_dropout)
        x_lstm = self.activation(x_lstm)
        output = self.lstm(x_lstm)[-1][0].squeeze(0)
        output = self.TrendNorm(output)

        return output
    

class myDLModel(nn.Module):
    """
    Apply a FeatureExtractor to each input column separately, concatenate, then predict.

    Inputs:
      - 'static':  (B, L, 1)
      - 'dynamic': (B, L, F_dynamic)
      - 'label':   (B, 1)

    Outputs:
      - 'prediction':  1D
      - 'loss'

    """
    def __init__(self, config):
        super().__init__()
        n_feats = len(config['used_features'])
        D = config['model_dim']

        self.feature_extractor_list = nn.ModuleList([FeatureExtractor(config) for _ in range(n_feats)])

        for model in self.feature_extractor_list:
            model.to(config['device'])

        self.prediction_layer = torch.nn.Linear(2 * D * n_feats, 1)
        self.loss_func = torch.nn.HuberLoss()

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        indicators = torch.cat([x['static'], x['dynamic']], dim=-1)
        feature_list = []

        for extractor, single in zip(self.feature_extractor_list,
                                      indicators.split(1, dim=-1)): 
            single = single.squeeze(-1)                    
            feature_list.append(extractor(single))

        features = torch.cat(feature_list, dim=-1)
        prediction_output = self.prediction_layer(features)
        loss_class = self.loss_func(prediction_output, x['label'])
        loss = loss_class

        return {'prediction': prediction_output,
                'loss': loss}
    
