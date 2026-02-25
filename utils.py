import matplotlib.pyplot as plt
import numpy as np
import torch
from data import inverse_transform
import shap

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_test_predictions(test_preds, test_loader, args, scaler_path=None):
    # Get experiment name
    exp_name = args.experiment_name if hasattr(args, 'experiment_name') else 'final'
    
    # Use provided scaler path or construct from experiment name with folder structure
    if scaler_path is None:
        scaler_path = f"Transforms/{exp_name}/labels_scaled.pkl"  # ← Added folder
    
    # Get actual test labels
    actual_labels = []
    for _, targets in test_loader:
        actual_labels.append(targets)
    actual_labels = torch.cat(actual_labels, dim=0).cpu().numpy()
    
    # Inverse transform
    inversed_actual = inverse_transform(actual_labels, scaler_path)
    
    # Get output names from config
    output_names = args.labels if hasattr(args, 'labels') else ['Output']
    
    # Plot predictions vs actual
    n_outputs = test_preds.shape[1] if len(test_preds.shape) > 1 else 1
    
    if n_outputs > 1:
        for i in range(n_outputs):
            plt.figure(figsize=(12, 6))
            plt.plot(inversed_actual[:, i], label='Actual', linewidth=2)
            plt.plot(test_preds[:, i], label='Predicted', linewidth=2, alpha=0.7)
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title(f'{output_names[i]}: Predictions vs Actual')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(inversed_actual, label='Actual', linewidth=2)
        plt.plot(test_preds, label='Predicted', linewidth=2, alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'{output_names[0] if output_names else "Output"}: Predictions vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def explain_model(model, data_loader, args, num_samples=100):
    """
    Generate SHAP explanations using KernelExplainer
    (Model-agnostic)
    """
    model.eval()
    
    # Get feature names
    feature_names = []
    if hasattr(args, 'features'):
        feature_names.extend(args.features)
    if hasattr(args, 'dummy_vars'):
        feature_names.extend(args.dummy_vars)
    
    # Collect background and test data
    background_data = []
    test_data = []
    
    for i, (inputs, _) in enumerate(data_loader):
        # Extract last timestep from sequences
        last_step = inputs[:, -1, :].cpu().numpy()
        
        if i == 0:
            background_data = last_step[:num_samples]
        
        test_data.append(last_step)
        
        if len(test_data) * inputs.shape[0] >= 20:
            break
    
    test_data = np.vstack(test_data)[:20]  # Use first 20 samples
    
    # Create prediction wrapper for SHAP
    def model_predict(x):
        """Wrapper function that takes 2D array and returns predictions"""
        # Convert numpy array to tensor
        x_tensor = torch.FloatTensor(x).to(args.device)
        
        # Add sequence dimension (batch, 1, features)
        # Since we're only using last timestep, sequence length = 1
        x_tensor = x_tensor.unsqueeze(1)
        
        with torch.no_grad():
            output = model(x_tensor).cpu().numpy()
        
        return output
    
    # Create KernelExplainer
    explainer = shap.KernelExplainer(model_predict, background_data)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(test_data, nsamples=100)

    return explainer, shap_values, test_data, feature_names


def plot_shap_summary(shap_values, test_data, feature_names, output_idx=0, output_name=None):
    """Plot SHAP summary - shows feature importance"""
    
    # For multi-output models
    n_outputs = shap_values.shape[1] // len(feature_names) if len(shap_values.shape) == 2 else 1
    n_features = len(feature_names)
    
    if n_outputs > 1:
        # Reshape and extract specific output
        shap_values_reshaped = shap_values.reshape(shap_values.shape[0], n_features, n_outputs)
        values = shap_values_reshaped[:, :, output_idx]
    else:
        values = shap_values
    
    title = f'SHAP Feature Importance - {output_name}' if output_name else f'SHAP Feature Importance - Output {output_idx}'
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(values, test_data, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_shap_bar(shap_values, test_data, feature_names, output_idx=0, output_name=None):
    """Plot SHAP bar chart - mean absolute SHAP values"""
    
    # For multi-output models
    n_outputs = shap_values.shape[1] // len(feature_names) if len(shap_values.shape) == 2 else 1
    n_features = len(feature_names)
    
    if n_outputs > 1:
        # Reshape and extract specific output
        shap_values_reshaped = shap_values.reshape(shap_values.shape[0], n_features, n_outputs)
        values = shap_values_reshaped[:, :, output_idx]
    else:
        values = shap_values
    
    title = f'SHAP Mean Importance - {output_name}' if output_name else f'SHAP Mean Importance - Output {output_idx}'
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, test_data, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_shap_waterfall(explainer, shap_values, test_data, feature_names, sample_idx=0, output_idx=0, output_name=None):
    """Plot SHAP waterfall - explains a single prediction"""
    
    # For multi-output models, SHAP concatenates outputs
    # Shape is (samples, features * n_outputs)
    n_outputs = len(explainer.expected_value) if hasattr(explainer.expected_value, '__len__') else 1
    n_features = len(feature_names)
    
    if n_outputs > 1:
        # Reshape from (samples, features * outputs) to (samples, features, outputs)
        shap_values_reshaped = shap_values.reshape(shap_values.shape[0], n_features, n_outputs)
        values = shap_values_reshaped[sample_idx, :, output_idx]
        base_value = explainer.expected_value[output_idx]
        data = test_data[sample_idx]
    else:
        values = shap_values[sample_idx]
        base_value = explainer.expected_value
        data = test_data[sample_idx]
    
    # Create explanation object
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=data,
        feature_names=feature_names
    )
    
    title = f'SHAP Waterfall - Sample {sample_idx}, {output_name}' if output_name else f'SHAP Waterfall - Sample {sample_idx}, Output {output_idx}'
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_shap_dependence(shap_values, test_data, feature_names, feature_idx, output_idx=0, output_name=None):
    """Plot SHAP dependence plot for a specific feature"""
    
    # Handle multi-output case
    if isinstance(shap_values, list):
        values = shap_values[output_idx]
    else:
        values = shap_values
    
    feature_name = feature_names[feature_idx]
    title = f'SHAP Dependence: {feature_name} - {output_name}' if output_name else f'SHAP Dependence: {feature_name}'
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_idx, values, test_data, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    def plot_residual_diagnostics(actual, pred, label, save_dir=None):
    actual    = np.array(actual).flatten()
    pred      = np.array(pred).flatten()
    residuals = actual - pred
    max_lags  = min(24, len(residuals) // 2 - 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Residual Diagnostics — {label}')

    # Residuals over time
    axes[0,0].plot(residuals)
    axes[0,0].axhline(0, color='red', linestyle='--')
    axes[0,0].set_title('Residuals Over Time')

    # Distribution
    axes[0,1].hist(residuals, bins=20, density=True, alpha=0.7)
    xmin, xmax = axes[0,1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    axes[0,1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r')
    axes[0,1].set_title('Distribution')

    # ACF / PACF
    plot_acf(residuals,  lags=max_lags, ax=axes[1,0], title='ACF')
    plot_pacf(residuals, lags=max_lags, ax=axes[1,1], title='PACF')

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/residual_diagnostics.png", dpi=150)
    plt.show()