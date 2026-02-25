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
class QuantileLoss(nn.Module):
    def __init__(self, q=0.5):
        super().__init__()
        self.q = q
    
    def forward(self, pred, actual):
        errors = actual - pred
        return torch.mean(torch.max(self.q * errors, (self.q - 1) * errors))
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, predictions, targets):
        # Per-label RMSE — same structure as MAPELoss
        mse_per_label = torch.mean((predictions - targets) ** 2, dim=0)  # shape: (n_labels,)
        rmse_per_label = torch.sqrt(mse_per_label + 1e-6)                # shape: (n_labels,)
        return torch.mean(rmse_per_label)   
        
class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8, per_label=False):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon
        self.per_label = per_label

    def forward(self, output, target):
        mape = torch.abs((target - output) / (target + self.epsilon)) * 100
        if self.per_label:
            return mape.mean(dim=0)   # shape: (n_labels,) — one MAPE per label
        return mape.mean()

class MADLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        pred_diff   = pred[1:]   - pred[:-1]
        actual_diff = actual[1:] - actual[:-1]
        sign_prod   = torch.sign(pred_diff * actual_diff)
        
        # 1 = wrong direction, 0 = correct direction
        dir_loss = torch.mean((1 - sign_prod) / 2 * torch.abs(actual_diff))
        
        mae = torch.mean(torch.abs(actual - pred))
        
        return self.alpha * dir_loss + (1 - self.alpha) * mae
class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Combines MSE with directional penalty
        alpha: weight for directional component (0-1)
              0 = pure MSE, 1 = pure directional
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        # MSE component
        mse_loss = self.mse(predictions, targets)
        
        # Directional component (penalize wrong direction)
        if predictions.shape[0] > 1:  # Need at least 2 samples
            pred_diff = predictions[1:] - predictions[:-1]
            target_diff = targets[1:] - targets[:-1]
            
            # Check if signs match (same direction)
            direction_correct = (pred_diff * target_diff) > 0
            direction_loss = 1 - direction_correct.float().mean()
        else:
            direction_loss = torch.tensor(0.0, device=predictions.device)
        
        return (1 - self.alpha) * mse_loss + self.alpha * direction_loss

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, smooth_window=5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model_state = None
        self.val_history = []
        self.smooth_window = smooth_window
        self._model_ref = None

    def early_stop(self, validation_loss, model=None):
        self.val_history.append(validation_loss)
        if len(self.val_history) >= self.smooth_window:
            smoothed = np.mean(self.val_history[-self.smooth_window:])
        else:
            smoothed = validation_loss

        if smoothed < self.min_validation_loss:
            self.min_validation_loss = smoothed
            self.counter = 0
            if model is not None:
                self.save_checkpoint(model)
        elif smoothed >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def save_checkpoint(self, model):
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

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
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "Data")
    
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

    start = pd.to_datetime(getattr(args, 'start_date', "1992-01-01"))
    btr_data = btr_data[btr_data.index >= start]
    macro_data = macro_data[macro_data.index >= start]
    dummy = dummy[dummy.index >= start]

    df = btr_data.join(macro_data, how="inner").join(dummy, how="inner")

    feature_cols = args.features if hasattr(args, 'features') else ['BIR', 'BOC', 'Other Offices',"Non-tax Revenues", "Expenditures", 'TotalTrade_PHPMN', 'NominalGDP_disagg', 'Pop_disagg']
    label_cols = args.labels if hasattr(args, 'labels') else ['BIR', 'BOC', 'Other Offices',"Non-tax Revenues", "Expenditures"]
    dummy_vars = args.dummy_vars if hasattr(args, 'dummy_vars') else ['COVID-19','TRAIN','CREATE','FIST','BIR_COMM']

    use_lags = getattr(args, 'use_lags', True)

    # ── Log transform ──
    log_transform = getattr(args, 'log_transform', False)
    skip_log_cols = getattr(args, 'skip_log_cols', ['Inflation', 'USDPHP'])
    
    if log_transform:
        cols_to_log = set(label_cols + feature_cols) - set(skip_log_cols)
        for col in cols_to_log:
            if col in df.columns:
                df[col] = np.log1p(df[col])
    # ── End log transform ──

    df = dt.add_seasonal_features(df)

    if use_lags:
        df = dt.add_lag_features(df, label_cols, args.lag_periods)

    feature_dfs = []
    
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

    X = pd.concat(feature_dfs, axis=1).values.copy()
    y = df[label_cols].values.copy()

    train_data, val_data, test_data = dt.split_data(X)
    train_labels, val_labels, test_labels = dt.split_data(y)
    
    cv_data = np.concatenate([train_data, val_data], axis=0)
    cv_labels = np.concatenate([train_labels, val_labels], axis=0)

    return {
        'cv_data': cv_data,
        'cv_labels': cv_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'input_size': cv_data.shape[1],
        'output_size': cv_labels.shape[1],
        'log_transform': log_transform,  
    }
def train_model(model, dataloader, device, optimizer, criterion, l1_lambda, scheduler=None):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()   # <-- step after each batch

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

def run(model, train_loader, val_loader, test_loader, args, fold=None, label_scaler=None):
    
    set_seed(args.seed if hasattr(args, 'seed') else 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch,
        eta_min=args.lr * 0.01)
    
    is_tuning = getattr(args, 'tuning_mode', False)
    is_cv = fold is not None
    use_early_stopping = getattr(args, 'early_stopping_patience', 0) > 0
    
    early_stopper = EarlyStopper(
        patience=args.early_stopping_patience,
        min_delta=getattr(args, 'early_stopping_min_delta', 0),
        smooth_window=getattr(args, 'smooth_window', 5)
    ) if use_early_stopping else None

    train_losses = []
    val_losses = []

    for e in range(args.epoch):
        warmup_lr(optimizer, args.lr, e + 1, 10)
        train_loss = train_model(model, train_loader, args.device, optimizer, 
                             args.train_criterion, args.l1_lambda)
        val_loss, _ = evaluate(model, val_loader, args.device, args.train_criterion, args)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if early_stopper:
            if early_stopper.early_stop(val_loss, model=model):
                if not is_tuning:
                    prefix = f"Fold {fold + 1} - " if is_cv else ""
                    print(f"{prefix}Early stopping triggered at epoch {e+1}/{args.epoch}")
                break

        if not is_tuning and ((e + 1) % 100 == 0 or e == 0):
            prefix = f"Fold {fold + 1} - " if is_cv else ""
            print(f"{prefix}Epoch {e+1}/{args.epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if early_stopper:
        early_stopper.load_best_model(model)
        if not is_tuning:
            print(f"Restored best model (smoothed val loss: {early_stopper.min_validation_loss:.4f})")

    _, test_preds = evaluate(model, test_loader, args.device, args.test_criterion, args)

    exp_name = getattr(args, 'experiment_name', 'default')

    if label_scaler is not None:
        inversed_test_preds = dt.inverse_transform(test_preds, scaler=label_scaler)
    else:
        scaler_path = f"Transforms/{exp_name}/labels_scaled_{fold}.pkl" if fold is not None else f"Transforms/{exp_name}/labels_scaled.pkl"
        inversed_test_preds = dt.inverse_transform(test_preds, load_path=scaler_path)

    actual_labels = torch.cat([targets for _, targets in test_loader], dim=0).cpu().numpy()
    if label_scaler is not None:
        inversed_actual = dt.inverse_transform(actual_labels, scaler=label_scaler)
    else:
        inversed_actual = dt.inverse_transform(actual_labels, load_path=scaler_path)

    if getattr(args, 'log_transform', False) and not getattr(args, 'tuning_mode', False):
        inversed_test_preds = np.expm1(np.clip(inversed_test_preds, None, 20))
        inversed_actual = np.expm1(np.clip(inversed_actual, None, 20))

    label_cols = getattr(args, 'labels', [])
    epsilon = 1e-8
    per_label_mape = np.mean(
        np.abs((inversed_actual - inversed_test_preds) / (inversed_actual + epsilon)) * 100,
        axis=0
    )
    
    per_label_mape_dict = {
        f"loss_{label_cols[i] if i < len(label_cols) else f'label_{i}'}": float(per_label_mape[i])
        for i in range(len(per_label_mape))
    }

    test_loss = torch.tensor(float(np.mean(per_label_mape)))
    if is_tuning:
        pred_std   = np.std(inversed_test_preds)
        actual_std = np.std(inversed_actual)
        if pred_std < 0.05 * actual_std:
            test_loss = torch.tensor(999.0)

    return test_loss, inversed_test_preds, inversed_actual, train_losses, val_losses, per_label_mape_dict

def crossval(data, labels, args):  # REMOVED: test_data, test_labels
    """
    Cross-validation for hyperparameter tuning.
    Evaluates on validation folds within the data.
    """
    n_splits = getattr(args, 'n_splits', 5)
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
        train_data_scaled, data_scaler   = dt.transform_data(train_data_fold)
        train_labels_scaled, label_scaler = dt.transform_data(train_labels_fold)
        val_data_scaled   = data_scaler.transform(val_data_fold)
        val_labels_scaled = label_scaler.transform(val_labels_fold)
        
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
        model = mod.GRUModel(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=output_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_attention_heads=getattr(args, 'num_attention_heads', 4),
            args=args
        ).to(args.device)
        
        # Train and evaluate on validation fold
        test_loss, test_preds, inversed_actual, train_losses, val_losses, per_label_mape_dict = run(
            model, 
            train_loader, 
            val_loader, 
            val_loader,  # Use val_loader for evaluation
            args,
            fold=fold,
            label_scaler=label_scaler
        )
        
        preds_np   = test_preds.cpu().numpy()   if torch.is_tensor(test_preds)   else np.array(test_preds)
        actuals_np = inversed_actual.cpu().numpy() if torch.is_tensor(inversed_actual) else np.array(inversed_actual)
        
        # Flatten if needed
        preds_flat   = preds_np.flatten()
        actuals_flat = actuals_np.flatten()

        pred_std   = np.std(preds_flat)
        actual_std = np.std(actuals_flat)
        
        # Directional accuracy
        pred_diff   = np.diff(preds_flat)
        actual_diff = np.diff(actuals_flat)
        dir_acc     = float(np.mean((pred_diff * actual_diff) > 0))
        
        # how well it predicts the top x% of values
        def mape_on_top_quantile(preds, actuals, q=0.90):
            threshold = np.quantile(actuals, q)
            mask = actuals >= threshold
            if mask.sum() == 0:
                return 0.0
            return float(np.mean(np.abs((actuals[mask] - preds[mask]) / (actuals[mask] + 1e-8))) * 100)
        
        peak_mape = mape_on_top_quantile(preds_flat, actuals_flat, q=0.90)
        if pred_std < 0.05 * actual_std:
            peak_mape = 999.0
            dir_acc   = 0.0
            test_loss = torch.tensor(999.0)
        
        # Store results
        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss.item() if torch.is_tensor(test_loss) else test_loss,
            'per_label_mape': per_label_mape_dict,
            'test_preds': test_preds,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'dir_acc':    dir_acc,       
            'peak_mape':  peak_mape,     
        })
        
        if not is_tuning:
            label_str = "  ".join([f"{k}: {v:.4f}" for k, v in per_label_mape_dict.items()])
            print(f"Fold {fold + 1} — Mean MAPE: {test_loss:.4f}  |  {label_str}")
    
    if not is_tuning:
        test_losses = [r['test_loss'] for r in fold_results]
        print(f"\n{'='*50}\nCross-Validation Results:\n{'='*50}")
        print(f"Mean MAPE: {np.mean(test_losses):.4f} (+/- {np.std(test_losses):.4f})")

        # Per-label summary across folds
        all_label_keys = fold_results[0]['per_label_mape'].keys()
        for k in all_label_keys:
            vals = [r['per_label_mape'][k] for r in fold_results]
            print(f"  {k}: {np.mean(vals):.4f} (+/- {np.std(vals):.4f})")
    
    return fold_results