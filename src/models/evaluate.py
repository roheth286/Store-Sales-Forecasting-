

import numpy as np
from sklearn.metrics import root_mean_squared_error


def compute_rmsle(y_true_log, y_pred_log):
    y_pred_log = np.clip(y_pred_log, 0, None)
    return root_mean_squared_error(y_true_log, y_pred_log)
