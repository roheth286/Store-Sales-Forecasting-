"""
Evaluation metrics module for Store Sales Forecasting.
"""

import numpy as np
from sklearn.metrics import root_mean_squared_error


def compute_rmsle(y_true_log, y_pred_log):
    """
    Computes Root Mean Squared Logarithmic Error (RMSLE).
    """
    y_pred_log = np.clip(y_pred_log, 0, None)
    return root_mean_squared_error(y_true_log, y_pred_log)
