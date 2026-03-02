import os
import itertools
import warnings
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sarimax import plot_target
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima.arima import auto_arima
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings('ignore')
# ── Metrics ────────────────────────────────────────────────
def compute_metrics(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    return {'mape': mape, 'rmse': rmse}


# ── Grid Search ────────────────────────────────────────────
def gridsearch(
    train,
    val,
    p_range,
    d_range,
    q_range,
    P_range=None,
    D_range=None,
    Q_range=None,
    s_range=None,
    top_n=5,
    n_jobs=1,
):
    train = np.array(train)
    val = np.array(val)

    orders = list(itertools.product(p_range, d_range, q_range))
    use_seasonal = all(arg is not None for arg in (P_range, D_range, Q_range, s_range))

    if use_seasonal:
        seasonal_orders = list(itertools.product(P_range, D_range, Q_range, s_range))
        candidates = [(order, seasonal) for order in orders for seasonal in seasonal_orders]
        print(f"  Fitting {len(candidates)} models (seasonal)...")
    else:
        candidates = [(order, None) for order in orders]
        print(f"  Fitting {len(candidates)} models...")

    val_reindexed = pd.Series(val, index=range(len(train), len(train) + len(val)))

    def _fit_candidate(candidate):
        order, seasonal = candidate
        try:
            if seasonal is not None:
                mod = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mod.fit(disp=False)
            else:
                mod = ARIMA(train, order=order)
                res = mod.fit()

            res_extended = res.append(val_reindexed, refit=False)
            preds = res_extended.predict(start=len(train), end=len(train) + len(val) - 1)
            mse = np.mean((val - np.array(preds)) ** 2)

            if seasonal is not None:
                return {'order': order, 'seasonal_order': seasonal, 'MSE': mse}
            return {'order': order, 'MSE': mse}
        except Exception:
            return None

    results = []
    if int(n_jobs) <= 1:
        for candidate in candidates:
            row = _fit_candidate(candidate)
            if row is not None:
                results.append(row)
    else:
        max_workers = min(int(n_jobs), len(candidates))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fit_candidate, candidate) for candidate in candidates]
            for future in as_completed(futures):
                row = future.result()
                if row is not None:
                    results.append(row)

    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError("No models converged.")
    return df.sort_values('MSE').head(top_n)

# ── Final Fit ──────────────────────────────────────────────
def fit_sarima(train, test, order, seasonal_order=None, walk_forward=False):
    train = np.array(train)
    test  = np.array(test)
    
    if walk_forward:
        # uses real residuals
        if seasonal_order is not None:
            res = SARIMAX(
                train,
                order=tuple(order),
                seasonal_order=tuple(seasonal_order),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        else:
            res = ARIMA(train, order=tuple(order)).fit()
        history = pd.Series(test, index=range(len(train), len(train) + len(test)))
        res_extended = res.append(history, refit=False)
        preds = res_extended.predict(start=len(train), end=len(train) + len(test) - 1)
        preds = np.array(preds)
    else:
        if seasonal_order is not None:
            res = SARIMAX(
                train,
                order=tuple(order),
                seasonal_order=tuple(seasonal_order),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        else:
            res = ARIMA(train, order=tuple(order)).fit()

        preds = res.forecast(steps=len(test))
        preds = np.array(preds)
    
    return preds, compute_metrics(test, preds)

def plot_target(df, target: str, figsize=(10, 5), color='green', linewidth=1):

    x = df.index
    y = df[target] 

    plt.figure(figsize=figsize)
    plt.title(f'Monthly {target} (2000 - 2025Q2)', fontsize=14)  
    plt.plot(x, y, color=color, linewidth=linewidth)

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.xlabel("Year")
    plt.ylabel(target) 
    plt.xticks(rotation=45, fontsize=10)

    plt.tight_layout()
    plt.show()

def multitarget_sarima(
    train_df,
    test_df,
    exog_train,
    exog_test,
    labels,
    bounds_map,
    seasonal_period=12,
    verbose=True,
    n_jobs=1,
    progress_every=50,
    ):
    best_rows = []
    all_rows = []
    forecasts_df = pd.DataFrame(index=test_df.index)
    fitted_models = {}

    def _run_one_label(label):
        if label not in bounds_map:
            raise ValueError(f"Missing bounds for label: {label}")

        cfg = bounds_map[label]
        required_keys = [
            "p_min", "p_max", "d_min", "d_max", "q_min", "q_max",
            "P_min", "P_max", "D_min", "D_max", "Q_min", "Q_max"
        ]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(f"Missing keys for {label}: {missing}")

        p_values = range(cfg["p_min"], cfg["p_max"] + 1)
        d_values = range(cfg["d_min"], cfg["d_max"] + 1)
        q_values = range(cfg["q_min"], cfg["q_max"] + 1)
        P_values = range(cfg["P_min"], cfg["P_max"] + 1)
        D_values = range(cfg["D_min"], cfg["D_max"] + 1)
        Q_values = range(cfg["Q_min"], cfg["Q_max"] + 1)

        param_grid = list(itertools.product(p_values, d_values, q_values))
        seasonal_grid = list(itertools.product(P_values, D_values, Q_values))
        total_candidates = len(param_grid) * len(seasonal_grid)

        best_aic = np.inf
        best_order = None
        best_seasonal = None
        best_fit = None

        loop_desc = f"{label} grid"
        use_tqdm = verbose and int(n_jobs) == 1
        loop_iter = tqdm(param_grid, desc=loop_desc, leave=False) if use_tqdm else param_grid
        report_progress = verbose and int(n_jobs) != 1 and progress_every is not None and int(progress_every) > 0
        attempted = 0

        if verbose:
            print(f"[{label}] candidates: {total_candidates}")

        label_trials = []

        for order in loop_iter:
            for seasonal in seasonal_grid:
                attempted += 1
                if report_progress and attempted % int(progress_every) == 0:
                    print(f"[{label}] progress: {attempted}/{total_candidates}")

                seasonal_order = (seasonal[0], seasonal[1], seasonal[2], seasonal_period)
                try:
                    model = SARIMAX(
                        train_df[label],
                        exog=exog_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fitted = model.fit(disp=False)
                    aic = fitted.aic

                    label_trials.append({
                        "Target": label,
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "aic": aic
                    })

                    if aic < best_aic:
                        best_aic = aic
                        best_order = order
                        best_seasonal = seasonal_order
                        best_fit = fitted
                except Exception:
                    continue

        if report_progress:
            print(f"[{label}] done: {attempted}/{total_candidates}, converged={len(label_trials)}")

        if best_fit is None:
            best_row = {
                "Target": label,
                "Best_Order": None,
                "Best_Seasonal_Order": None,
                "Best_AIC": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE_pct": np.nan
            }
            return best_row, label_trials, None, None

        pred = best_fit.get_forecast(steps=len(test_df), exog=exog_test).predicted_mean
        pred = pred.reindex(test_df.index)
        actual = test_df[label]

        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        denom = np.where(actual != 0, actual, np.nan)
        mape = np.nanmean(np.abs((actual - pred) / denom)) * 100

        best_row = {
            "Target": label,
            "Best_Order": best_order,
            "Best_Seasonal_Order": best_seasonal,
            "Best_AIC": best_aic,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_pct": mape
        }

        return best_row, label_trials, pred, best_fit

    if int(n_jobs) == 1:
        for label in labels:
            best_row, label_trials, pred, best_fit = _run_one_label(label)
            best_rows.append(best_row)
            all_rows.extend(label_trials)
            if pred is not None:
                forecasts_df[label] = pred
            if best_fit is not None:
                fitted_models[label] = best_fit
    else:
        max_workers = min(int(n_jobs), len(labels))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_label = {executor.submit(_run_one_label, label): label for label in labels}
            completed_iter = as_completed(future_to_label)
            if verbose:
                completed_iter = tqdm(
                    completed_iter,
                    total=len(future_to_label),
                    desc="Labels completed",
                    leave=False,
                )

            for future in completed_iter:
                label = future_to_label[future]
                best_row, label_trials, pred, best_fit = future.result()
                best_rows.append(best_row)
                all_rows.extend(label_trials)
                if pred is not None:
                    forecasts_df[label] = pred
                if best_fit is not None:
                    fitted_models[label] = best_fit

    best_models_df = pd.DataFrame(best_rows).set_index("Target")
    all_trials_df = pd.DataFrame(all_rows).sort_values(["Target", "aic"]).reset_index(drop=True)
    return best_models_df, all_trials_df, forecasts_df, fitted_models

