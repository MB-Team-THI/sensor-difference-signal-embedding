import torch
import numpy as np
from src.utils.exp_functions import log_function 


def norm_error_signal(error_signal_batch_padded, error_signal_features, norm_max_values):
    # Normalize the batched input signal based on the max values within the dataset
    # Each feature [error_signal_features] will have a different max-value [norm_max_values]
    # The range of the normalized error signal will be -1 to 1

    for error_signal in error_signal_batch_padded:
        for idx, key in enumerate(error_signal_features):
            error_signal[:, idx] = error_signal[:, idx] / norm_max_values[key]

    return error_signal_batch_padded



def norm_error_signal_logarithmic(error_signal_batch_padded, error_signal_features, norm_max_values):      

    for error_signal in error_signal_batch_padded:
        for idx, key in enumerate(error_signal_features):
            neg_values = torch.tensor([-1 if x < 0 else 1 for x in error_signal[:, idx]]).to(error_signal.device)

            x     = (abs(error_signal[:, idx])+1).cpu()
            base  = norm_max_values[key] + 1
            x_log = np.emath.logn(base, x)

            x_log_tensor = torch.tensor(x_log).to(error_signal.device)

            error_signal[:, idx] = neg_values * x_log_tensor


    return error_signal_batch_padded
            