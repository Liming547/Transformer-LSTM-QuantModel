import numpy as np
import pandas as pd
import copy
import torch
import yaml  
from tqdm import tqdm
from pathlib import Path 
from utils import get_dataloader, get_dataloader_dask, validate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dlmodel import myDLModel


def load_config(cfg_path: str = "config.yaml"):
    """
        Read the YAML config and discover input files.

        Parameters
        ----------
        cfg_path : str
            Path to the configuration file.

        Returns
        -------
        data : pandas.DataFrame or dict
            If there is only one .pkl file, this is the loaded DataFrame.
            If there are multiple .pkl files, this is a dict like
            {'files': List[Path], 'config': Dict} for the Dask path.
        config : dict
            hyperparameters.
    """
    with Path(cfg_path).open() as f:
        cfg_yaml = yaml.safe_load(f)

    config = cfg_yaml["params"]
    config['feature_num']     = len(config['used_features'])
    config['lr']              = float(config['lr'])
    config['lookback']        = int(config['lookback'])
    config['batch_size']      = int(config['batch_size'])
    config['epoch_num']       = int(config['epoch_num'])
    config['save_model_path'] = cfg_yaml["save_model_path"]

    market_data_path  = cfg_yaml["train_data_path"]
    files = sorted(Path(market_data_path).glob("*.pkl"))
    
    if len(files) > 1:
        return {'files': files, 'config': config}, config
    else:
        data = pd.read_pickle(files[0])
        return data, config


def train(data = None, config = None):
    """
        Run the training loop.

        Behavior
        --------
        If there are multiple input files (Dask path), build loaders with
        `get_dataloader_dask`. Otherwise, use `get_dataloader` on the single
        DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            Either a single DataFrame (classic path) or the Dask dict
            {'files': List[Path], 'config': Dict}.
        config : dict
            Training/eval hyperparameters and paths.

        Outputs
        ------------
        Saves the best model checkpoint to `config['save_model_path']`.
    """
    torch.manual_seed(3529)
    np.random.seed(3529)


    if isinstance(data, dict) and 'files' in data:
        # Dask multi-file path
        files = data['files']
        dataloader_train, dataloader_val, _ = get_dataloader_dask(files,
                                                                lookback=config['lookback'],
                                                                used_features=config['used_features'],
                                                                batch_size=config['batch_size'],
                                                                chunk_size=config['chunk_size'],
                                                                device=config['device'])
    else:
        dataloader_train, dataloader_val, _ = get_dataloader(config['lookback'], 
                                                             config['used_features'],
                                                             config['batch_size'], 
                                                             config['device'],
                                                             data = data)
    


    model = myDLModel(config).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=1)

    best_val_loss = float('inf')
    best_model_weight = copy.deepcopy(model.state_dict())

    for epoch_index in range(config['epoch_num']):
        print('epoch:', epoch_index)

        model.train()
        loss_all = []
        for _, batch in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            batch_to_device = {k: (v.to(config['device']) if torch.is_tensor(v) else v) for k, v in batch.items()}
            output = model(batch_to_device)
            loss = output['loss']
            loss_all.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


        val_loss = validate(dataloader_val, model)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weight = copy.deepcopy(model.state_dict())
            print(f'val_loss={best_val_loss:.6f}.')


    torch.save(best_model_weight, config['save_model_path'])




if __name__ == '__main__':
    data, config = load_config("config.yaml")
    train(data, config)