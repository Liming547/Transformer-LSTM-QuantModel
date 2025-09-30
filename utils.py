import numpy as np
import pandas as pd
from tqdm import tqdm
from dask import delayed, compute
import dask.array as da
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Iterable, List, Tuple


# ----------------------------- Define Dataset ----------------------------- #

class myDataset(Dataset):
    """
        Time-series dataset with dynamic & static features.

        Attributes:
            data_dynamic (np.ndarray): Dynamic features of shape (N, L, num_dyn),
                where N = samples, L = lookback/sequence length, num_dyn = # dynamic features.
            data_static (np.ndarray): Static feature of shape (N, L, 1).
            label (np.ndarray): Targets of shape (N, 1).

    """
    def __init__(self, data_dynamic, data_static, label, device='cuda'):
        self.data_dynamic = data_dynamic
        self.data_static = data_static
        self.device = device
        self.label = label

    def __len__(self):
        return len(self.data_dynamic)

    def __getitem__(self, idx):
        static = torch.from_numpy(self.data_static[idx].astype(np.float32))
        dynamic = torch.from_numpy(self.data_dynamic[idx].astype(np.float32))
        label = torch.tensor(self.label[idx], dtype=torch.float32)

        return {'static': static,
                'dynamic': dynamic,
                'label': label}

# ----------------------- Preprocess (Feature Engineering) ------------------------- #

def preprocess_data(data: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
        preprocess the features
    """
    #data['feature1'] = 
    #data['feature2'] = 
    #data['feature3'] = 
    # ...
    return data

def create_sequences(data: np.ndarray, target: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
        create overlapping sliding windows for the time-series data, which are used in training.
        we use x[i-lookback:i] to predict y[i].
    """
    X, y = [], []
    
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(target.iloc[i])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    return X, y 

# ----------------------- Construct Dataloader ------------------------- #

def load_and_engineer(path: str, lookback: int) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = preprocess_data(df, lookback) 
    df.dropna(inplace=True)
    return df

def make_sequences_numpy(df: pd.DataFrame, used_features: List[str], lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create (X, y) using numpy for ONE dataframe."""
    data_scaled = df[used_features].to_numpy(dtype=np.float32)
    X, y = create_sequences(data_scaled, df['Returns'].to_numpy(dtype=np.float32), lookback=lookback)
    return X, y


def get_dataloader(    
        lookback: int = 21,
        used_features: List[str] = None,
        batch_size: int = 64,
        device: str = "cuda",
        data: pd.DataFrame = None,
        )  -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
        Build train/val/test loaders (single input .pkl file)
    """

    # data preprocess, normalization, and create a sequence of data for training
    data = preprocess_data(data, lookback)
    data_clean = data.dropna()
    data_scaled = StandardScaler().fit_transform(data_clean[used_features])
    X, y = create_sequences(data_scaled, data['Returns'], lookback=lookback)
 
    # split training/val/test data
    total_len = len(X)
    train_len = int(total_len * 0.7)
    X_train, X_val, X_test = X[:train_len], \
                             X[train_len:int(0.5 * (train_len + total_len))], \
                             X[int(0.5 * (train_len + total_len)):]
    
    y_train, y_val, y_test = y[:train_len], \
                             y[train_len:int(0.5 * (train_len + total_len))], \
                             y[int(0.5 * (train_len + total_len)):]
    
    # split dynamic and static features
    num_dyn = len(used_features) - 1
    X_train_dynamic = X_train[:, :, :num_dyn]
    X_train_static  = X_train[:, :, num_dyn].reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_dynamic   = X_val[:, :, :num_dyn]
    X_val_static    = X_val[:, :, num_dyn].reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test_dynamic  = X_test[:, :, :num_dyn]
    X_test_static   = X_test[:, :, num_dyn].reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # construct dataloader
    dataset_train = myDataset(X_train_dynamic, X_train_static,
                                  y_train, device=device)
    dataset_val = myDataset(X_val_dynamic, X_val_static,
                                y_val, device=device)
    dataset_test = myDataset(X_test_dynamic, X_test_static,
                                 y_test, device=device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    return dataloader_train, dataloader_val, dataloader_test

def get_dataloader_dask(
        files: Iterable[str],
        lookback: int = 21,
        used_features: List[str] = None,
        batch_size: int = 64,
        chunk_size: int = 10000,
        device: str = "cuda",
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    """
    Build train/val/test loaders (multiple input .pkl files):
      1) load and do feature engineering each file in parallel using dask.delayed
      2) fit a global scaler on a dask.array
      3) creating sequences per file in parallel
      4) concatenating numpy arrays
    """
    assert len(files) > 0, "No input files found"


    # create delayed jobs
    delayed_dfs = [delayed(load_and_engineer)(str(p), lookback) for p in files]
    dfs = compute(*delayed_dfs)  

    # Global scaling on all data
    arrays = [da.from_array(df[used_features].to_numpy(dtype=np.float32), chunks=(chunk_size, len(used_features)))
              for df in dfs]
    all_data = da.concatenate(arrays, axis=0)
    scaler = DaskStandardScaler()   
    scaler.fit(all_data)               

    # Transform each file separately
    transformed_data = []
    for df in dfs:
        Xa = da.from_array(df[used_features].to_numpy(dtype=np.float32),
                           chunks=(chunk_size, len(used_features)))
        Xa_scaled = scaler.transform(Xa)
        data_scaled = df.copy()
        data_scaled[used_features] = Xa_scaled.compute()
        transformed_data.append(data_scaled)

    # Create sequences per file 
    delayed_Xy = [delayed(make_sequences_numpy)(data_scaled, used_features, lookback)
                  for data_scaled in transformed_data]
    Xy_list = compute(*delayed_Xy)  # list of (X, y) tuples

    # Concatenate 
    X = np.concatenate([xy[0] for xy in Xy_list], axis=0)
    y = np.concatenate([xy[1] for xy in Xy_list], axis=0)

    # split training/val/test data
    total_len = len(X)
    train_len = int(total_len * 0.7)
    X_train, X_val, X_test = X[:train_len], X[train_len:int(0.5 * (train_len + total_len))], X[int(0.5 * (train_len + total_len)):]
    y_train, y_val, y_test = y[:train_len], y[train_len:int(0.5 * (train_len + total_len))], y[int(0.5 * (train_len + total_len)):]

    # split dynamic and static features
    num_dyn = len(used_features) - 1
    X_train_dynamic = X_train[:, :, :num_dyn]
    X_train_static  = X_train[:, :, num_dyn].reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_dynamic   = X_val[:,   :, :num_dyn]
    X_val_static    = X_val[:,   :, num_dyn].reshape(X_val.shape[0],   X_val.shape[1],   1)
    X_test_dynamic  = X_test[:,  :, :num_dyn]
    X_test_static   = X_test[:,  :, num_dyn].reshape(X_test.shape[0],  X_test.shape[1],  1)

    # construct dataloader
    dataset_train = myDataset(X_train_dynamic, X_train_static, y_train, device=device)
    dataset_val   = myDataset(X_val_dynamic,   X_val_static,   y_val,   device=device)
    dataset_test  = myDataset(X_test_dynamic,  X_test_static,  y_test,  device=device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=(device=='cuda'), persistent_workers=True)
    dataloader_val   = DataLoader(dataset_val,   batch_size=batch_size, shuffle=False,
                                  num_workers=2, pin_memory=(device=='cuda'), persistent_workers=True)
    dataloader_test  = DataLoader(dataset_test,  batch_size=batch_size, shuffle=False,
                                  num_workers=2, pin_memory=(device=='cuda'), persistent_workers=True)


    return dataloader_train, dataloader_val, dataloader_test



def validate(dataloader_val: DataLoader, model: torch.nn.Module) -> float:
    """
        Evaluate on the validation set and return mean loss.
    """
    loss_all = []
    predictions_val = []
    model.eval()
    for _, data in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):
        device = next(model.parameters()).device
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
        output = model(data)
        predictions_val.append(output['prediction'].cpu().detach().numpy())
        loss_all.append(output['loss'].item())
    
    return np.mean(loss_all)

