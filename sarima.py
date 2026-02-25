import os
import itertools
import warnings
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')
# ── Metrics ────────────────────────────────────────────────
def compute_metrics(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    return {'mape': mape, 'rmse': rmse}


# ── Grid Search ────────────────────────────────────────────
def gridsearch(train, val, p_range, d_range, q_range, top_n=5):
    from statsmodels.tsa.arima.model import ARIMA

    train = np.array(train)
    val   = np.array(val)

    orders  = list(itertools.product(p_range, d_range, q_range))
    results = []
    print(f"  Fitting {len(orders)} models...")

    for i, order in enumerate(orders):
        try:
            mod = ARIMA(train, order=order)
            res = mod.fit()
            val_reindexed = pd.Series(val, index=range(len(train), len(train) + len(val)))
            res_extended  = res.append(val_reindexed, refit=False)
            preds         = res_extended.predict(start=len(train), end=len(train) + len(val) - 1)

            mse = np.mean((val - np.array(preds)) ** 2)
            results.append({'order': order, 'MSE': mse})
        except Exception:
            continue

    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError("No models converged.")
    return df.sort_values('MSE').head(top_n)

# ── Final Fit ──────────────────────────────────────────────
def fit_sarima(train, test, order, walk_forward=False):
    train = np.array(train)
    test  = np.array(test)
    
    if walk_forward:
        # uses real residuals
        res = ARIMA(train, order=tuple(order)).fit()
        history = pd.Series(test, index=range(len(train), len(train) + len(test)))
        res_extended = res.append(history, refit=False)
        preds = res_extended.predict(start=len(train), end=len(train) + len(test) - 1)
        preds = np.array(preds)
    else:
        
        res = ARIMA(train, order=tuple(order)).fit()
        preds = res.forecast(steps=len(test))
        preds = np.array(preds)
    
    return preds, compute_metrics(test, preds)