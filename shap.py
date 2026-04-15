import matplotlib.pyplot as plt
import numpy as np
import torch
from data import inverse_transform
import shap
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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


def plot_qq(actual_dict, pred_dict, label_cols, save_path=None):
    fig, axes = plt.subplots(1, len(label_cols), figsize=(5 * len(label_cols), 4))
    if len(label_cols) == 1:
        axes = [axes]

    for ax, label in zip(axes, label_cols):
        residuals = np.array(actual_dict[label]).flatten() - np.array(pred_dict[label]).flatten()
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot — {label}')
        ax.get_lines()[1].set_color('red')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

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

def plot_test_predictions(test_preds, test_actuals, label, save_dir=None):
    test_preds   = np.array(test_preds).flatten()
    test_actuals = np.array(test_actuals).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(test_actuals, label='Actual',    linewidth=2)
    plt.plot(test_preds,   label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'{label}: Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/predictions.png", dpi=150)
    plt.show()

def explain_model(models, data_loader, args, num_samples=100):
    """Generate SHAP explanations for TCN using full sequence inputs."""
    for m in models:
        m.eval()

    # Build base feature names from args
    base_feature_names = []
    if hasattr(args, 'features'):
        base_feature_names.extend(args.features)
    if hasattr(args, 'labels') and hasattr(args, 'lag_periods'):
        for label in args.labels:
            for lag in args.lag_periods:
                base_feature_names.append(f'{label}_lag_{lag}')
    if hasattr(args, 'dummy_vars'):
        base_feature_names.extend(args.dummy_vars)
    if getattr(args, 'use_seasonal', False):
        base_feature_names.extend(['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
                                   'is_tax_season', 'is_year_end'])

    # Collect full sequence tensors and flatten for SHAP
    background_data = None
    test_batches = []
    n_features = None
    seq_len = None
    for i, (inputs, _) in enumerate(data_loader):
        arr = inputs.cpu().numpy()  # [batch, features, seq_len]
        if n_features is None:
            n_features, seq_len = arr.shape[1], arr.shape[2]
        flat = arr.reshape(arr.shape[0], -1)
        if background_data is None:
            background_data = flat[:num_samples]
        test_batches.append(flat)
        if sum(b.shape[0] for b in test_batches) >= 20:
            break
    test_data = np.vstack(test_batches)[:20]

    # Ensure feature-name count matches flattened input width
    if not base_feature_names or len(base_feature_names) != n_features:
        base_feature_names = [f'feature_{i}' for i in range(n_features)]
    feature_names = []
    for fname in base_feature_names:
        for t in range(seq_len):
            feature_names.append(f'{fname}_t{t}')

    # prediction wrapper
    def model_predict(x):
        x_tensor = torch.FloatTensor(x).to(args.device).reshape(-1, n_features, seq_len)
        preds = []
        with torch.no_grad():
            for m in models:
                preds.append(m(x_tensor).cpu().numpy())
        return np.mean(preds, axis=0)

    explainer = shap.KernelExplainer(model_predict, background_data)
    shap_values = explainer.shap_values(test_data, nsamples=100)

    return explainer, shap_values, test_data, feature_names


def plot_shap_summary(shap_values, test_data, feature_names, output_name=None):
    """Plot SHAP summary - shows feature importance."""
    title = f'SHAP Feature Importance - {output_name}' if output_name else 'SHAP Feature Importance'
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, test_data, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_shap_bar(shap_values, test_data, feature_names, output_name=None):
    """Plot SHAP bar chart - mean absolute SHAP values."""
    title = f'SHAP Mean Importance - {output_name}' if output_name else 'SHAP Mean Importance'
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_data, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_shap_waterfall(explainer, shap_values, test_data, feature_names, sample_idx=0, output_name=None):
    """Plot SHAP waterfall - explains a single prediction."""
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=test_data[sample_idx],
        feature_names=feature_names,
    )
    title = f'SHAP Waterfall - Sample {sample_idx}, {output_name}' if output_name else f'SHAP Waterfall - Sample {sample_idx}'
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_shap_dependence(shap_values, test_data, feature_names, feature_idx, output_name=None):
    """Plot SHAP dependence plot for a specific feature."""
    feature_name = feature_names[feature_idx]
    title = f'SHAP Dependence: {feature_name} - {output_name}' if output_name else f'SHAP Dependence: {feature_name}'
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_idx, shap_values, test_data, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show() 

def explain_model(models, data_loader, args, num_samples=100):
    """Generate SHAP explanations for TCN using full sequence inputs."""
    for m in models:
        m.eval()

    base_feature_names = []
    if hasattr(args, 'features'):
        base_feature_names.extend(args.features)
    if hasattr(args, 'labels') and hasattr(args, 'lag_periods'):
        for label in args.labels:
            for lag in args.lag_periods:
                base_feature_names.append(f'{label}_lag_{lag}')
    if hasattr(args, 'dummy_vars'):
        base_feature_names.extend(args.dummy_vars)
    if getattr(args, 'use_seasonal', False):
        base_feature_names.extend([
            'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
            'is_tax_season', 'is_year_end'
        ])

    background_data = None
    test_batches = []
    n_features = None
    seq_len = None

    for i, (inputs, _) in enumerate(data_loader):
        arr = inputs.cpu().numpy()  # [batch, features, seq_len]
        if n_features is None:
            n_features, seq_len = arr.shape[1], arr.shape[2]
        flat = arr.reshape(arr.shape[0], -1)
        if background_data is None:
            background_data = flat[:num_samples]
        test_batches.append(flat)
        if sum(b.shape[0] for b in test_batches) >= 20:
            break

    test_data = np.vstack(test_batches)[:20]

    if not base_feature_names or len(base_feature_names) != n_features:
        base_feature_names = [f'feature_{i}' for i in range(n_features)]

    feature_names = []
    for fname in base_feature_names:
        for t in range(seq_len):
            feature_names.append(f'{fname}_t{t}')

    def model_predict(x):
        x_tensor = torch.FloatTensor(x).to(args.device).reshape(-1, n_features, seq_len)
        preds = []
        with torch.no_grad():
            for m in models:
                preds.append(m(x_tensor).cpu().numpy())
        return np.mean(preds, axis=0)

    explainer = shap.KernelExplainer(model_predict, background_data)
    shap_values = explainer.shap_values(test_data, nsamples=100)

    return explainer, shap_values, test_data, feature_names

def plot_shap_waterfall(explainer, shap_values, test_data, feature_names, sample_idx=0, output_name=None):
    """Plot SHAP waterfall for one sample, robust to SHAP output shapes."""
    sv = shap_values
    if isinstance(sv, list):
        sv = sv[0]

    sv = np.asarray(sv)
    row = np.asarray(sv[sample_idx]).squeeze()

    if row.ndim != 1:
        row = row.reshape(-1)

    base_val = np.asarray(explainer.expected_value).squeeze()
    if np.ndim(base_val) > 0:
        base_val = float(np.ravel(base_val)[0])
    else:
        base_val = float(base_val)

    sample_data = np.asarray(test_data[sample_idx]).reshape(-1)

    explanation = shap.Explanation(
        values=row,
        base_values=base_val,
        data=sample_data,
        feature_names=feature_names,
    )

    title = (
        f"SHAP Waterfall - Sample {sample_idx}, {output_name}"
        if output_name else f"SHAP Waterfall - Sample {sample_idx}"
    )
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()