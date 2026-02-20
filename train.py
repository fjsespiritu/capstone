import pandas as pd
import numpy as np
import random  
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model as mod
import data as dt

from sklearn.model_selection import TimeSeriesSplit

class Arguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, predictions, targets):       
        return torch.sqrt(self.mse(predictions, targets) + 1e-6)
        
class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon  # Small constant to prevent division by zero

    def forward(self, output, target):
        return torch.mean(torch.abs((target - output) / (target + self.epsilon)))*100

def set_seed(seed=None):
    if seed is None:
        seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Remove CUDA-specific settings for Mac
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    os.environ['PYTHONHASHSEED'] = str(seed)

def load_dataset(args):
    """Load and prepare all datasets"""
    
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "Data")
    
    # Load raw data files with absolute paths
    btr_data = pd.read_csv(os.path.join(data_dir, "cordata.csv"))
    btr_data = btr_data.set_index("Date")

    macro_data = pd.read_csv(os.path.join(data_dir, "disaggregated.csv")).fillna(0)
    macro_data = macro_data.rename(columns={'Unnamed: 3': 'Date'}).set_index('Date')

    dummy = pd.read_csv(os.path.join(data_dir, "dummy.csv")).fillna(0)
    dummy = dummy.set_index("Date")

    btr_data.index = pd.to_datetime(btr_data.index)
    macro_data.index = pd.to_datetime(macro_data.index)
    dummy.index = pd.to_datetime(dummy.index)
    
    btr_data = btr_data.sort_index()
    macro_data = macro_data.sort_index()
    dummy = dummy.sort_index()

    # Filter by date e
    start = pd.to_datetime("1992-01-01")
    btr_data = btr_data[btr_data.index >= start]
    macro_data = macro_data[macro_data.index >= start]
    dummy = dummy[dummy.index >= start]

    # Join data
    df = btr_data.join(macro_data, how="inner").join(dummy, how="inner")

     # Extract configuration
    feature_cols = args.features if hasattr(args, 'features') else ['BIR', 'BOC', 'Other Offices',"Non-tax Revenues", "Expenditures", 'TotalTrade_PHPMN', 'NominalGDP_disagg', 'Pop_disagg']
    label_cols = args.labels if hasattr(args, 'labels') else ['BIR', 'BOC', 'Other Offices',"Non-tax Revenues", "Expenditures"]
    dummy_vars = args.dummy_vars if hasattr(args, 'dummy_vars') else ['COVID-19','TRAIN','CREATE','FIST','BIR_COMM']

    use_lags = getattr(args, 'use_lags', True)

    df = dt.add_seasonal_features(df)

    if use_lags:
        df = dt.add_lag_features(df, label_cols, args.lag_periods)

    
    feature_dfs = []
    
    # Add main features
    for col in feature_cols:
        if col in df.columns:
            feature_dfs.append(df[[col]])
        else:
            print(f"Warning: Feature '{col}' not found in data")

    if use_lags:
        lag_cols = [col for col in df.columns if '_lag_' in col]
        if lag_cols:
            feature_dfs.append(df[lag_cols])


    if dummy_vars:
        available_dummies = [d for d in dummy_vars if d in df.columns]
    
        if available_dummies:
            feature_dfs.append(df[available_dummies])  
            
    seasonal_cols = ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 
                     'is_tax_season', 'is_year_end']
    
    use_seasonal = getattr(args, 'use_seasonal', True)
    if use_seasonal:
        available_seasonal = [col for col in seasonal_cols if col in df.columns]

        if available_seasonal:
            feature_dfs.append(df[available_seasonal])

    # Create X and y
    X = pd.concat(feature_dfs, axis=1).values.copy()

    y = df[label_cols].values.copy()

    # Split into train/val/test
    train_data, val_data, test_data = dt.split_data(X)
    train_labels, val_labels, test_labels = dt.split_data(y)
    
    # Combine train + val for cross-validation
    cv_data = np.concatenate([train_data, val_data], axis=0)
    cv_labels = np.concatenate([train_labels, val_labels], axis=0)

    # Return everything needed
    return {
        'cv_data': cv_data,
        'cv_labels': cv_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'input_size': cv_data.shape[1],
        'output_size': cv_labels.shape[1]
    }

def train_model(model, dataloader, device, optimizer, criterion, l1_lambda):

    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0

@torch.no_grad()
def evaluate(model, dataloader, device, criterion, args):

    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)


        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        all_preds.append(outputs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return avg_loss, all_preds

def warmup_lr(optimizer, base_lr, epoch, warmup_epochs):
    """Apply learning rate warmup based on epoch number"""
    if epoch <= warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def run(model, train_loader, val_loader, test_loader, args, fold=None):
    """
    Train and evaluate model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        args: Arguments object containing all hyperparameters
        fold: Optional fold number for cross-validation
    """
    
    set_seed(args.seed if hasattr(args, 'seed') else 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=args.factor, 
        patience=args.patience
    )
    
    train_losses = []
    val_losses = []
    is_tuning = hasattr(args, 'tuning_mode') and args.tuning_mode
    for e in range(args.epoch):

        warmup_lr(optimizer, args.lr, e + 1, 10)

        train_loss = train_model(model, train_loader, args.device, optimizer, args.train_criterion, args.l1_lambda)
        val_loss, _ = evaluate(model, val_loader, args.device, args.train_criterion, args)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if not is_tuning and ((e + 1) % 100 == 0 or e == 0):
            fold_prefix = f"Fold {fold + 1} - " if fold is not None else ""
            print(f"{fold_prefix}Epoch {e+1}/{args.epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    _, test_preds = evaluate(model, test_loader, args.device, args.test_criterion, args)

    # Inverse transform using fold-specific or default scaler
    exp_name = args.experiment_name if hasattr(args, 'experiment_name') else 'default'
    
    if fold is not None:
        # Cross-validation run
        scaler_name = f"Transforms/{exp_name}/labels_scaled_{fold}.pkl"
    else:
        # Final model run
        scaler_name = f"Transforms/{exp_name}/labels_scaled.pkl"
    
    inversed_test_preds = dt.inverse_transform(test_preds, scaler_name)
    
    # Get corresponding actual labels from test_loader 
    actual_labels = []
    for _, targets in test_loader:
        actual_labels.append(targets)
    actual_labels = torch.cat(actual_labels, dim=0).cpu().numpy()
    
    # Inverse transform actual labels 
    inversed_actual = dt.inverse_transform(actual_labels, scaler_name)
    
    # Calculate test loss
    test_loss = args.test_criterion(
        torch.tensor(inversed_test_preds), 
        torch.tensor(inversed_actual)
    )
    
    return test_loss, inversed_test_preds, train_losses, val_losses

def crossval(data, labels, args, n_splits=5):  # REMOVED: test_data, test_labels
    """
    Cross-validation for hyperparameter tuning.
    Evaluates on validation folds within the data.
    """
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    
    exp_name = args.experiment_name if hasattr(args, 'experiment_name') else 'default'
    is_tuning = hasattr(args, 'tuning_mode') and args.tuning_mode
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
        if not is_tuning:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{n_splits}")
            print(f"{'='*50}")
        
        # Split data for this fold
        train_data_fold = data[train_idx]
        train_labels_fold = labels[train_idx]
        val_data_fold = data[val_idx]
        val_labels_fold = labels[val_idx]
        
        # Scale data
        set_seed(1)
        train_data_scaled = dt.transform_data(train_data_fold, f"Transforms/{exp_name}/train_scaled_{fold}.pkl")
        train_labels_scaled = dt.transform_data(train_labels_fold, f"Transforms/{exp_name}/labels_scaled_{fold}.pkl")
        val_data_scaled = dt.transform_data(val_data_fold, f"Transforms/{exp_name}/train_scaled_{fold}.pkl")
        val_labels_scaled = dt.transform_data(val_labels_fold, f"Transforms/{exp_name}/labels_scaled_{fold}.pkl")
        
        input_size = train_data_scaled.shape[1]
        output_size = train_labels_scaled.shape[1]
        
        # Create datasets
        train_dataset = dt.TimeSeriesDataset(train_data_scaled, train_labels_scaled, seq_len=args.seq_len)
        val_dataset = dt.TimeSeriesDataset(val_data_scaled, val_labels_scaled, seq_len=args.seq_len)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create fresh model for this fold
        set_seed(1)
        model = mod.TCNRegressor(
            input_size=input_size,
            num_channels=args.num_channels,
            output_size=output_size,
            kernel_size=args.kernel_size,
            dropout=args.dropout
        ).to(args.device)
        
        # Train and evaluate on validation fold
        test_loss, test_preds, train_losses, val_losses = run(
            model, 
            train_loader, 
            val_loader, 
            val_loader,  # Use val_loader for evaluation
            args,
            fold=fold
        )
        
        # Store results
        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss.item() if torch.is_tensor(test_loss) else test_loss,
            'test_preds': test_preds,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        })
        
        if not is_tuning:
            print(f"Fold {fold + 1} Val Loss: {test_loss:.4f}")
    
    if not is_tuning:
        test_losses = [r['test_loss'] for r in fold_results]
        print(f"\n{'='*50}")
        print(f"Cross-Validation Results:")
        print(f"{'='*50}")
        print(f"Mean Val Loss: {np.mean(test_losses):.4f} (+/- {np.std(test_losses):.4f})")
        print(f"Min Val Loss: {np.min(test_losses):.4f}")
        print(f"Max Val Loss: {np.max(test_losses):.4f}")
    
    return fold_results