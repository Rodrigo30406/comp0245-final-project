"""Offline errors: physical per-output errors and the episode- and group-balanced normalised error (NRMSE).

The primary offline metric is the balanced NRMSE: per episode, the RMSE of each output divided by the standard deviation
of that output over the training labels, averaged half over the two wheels and half over the seven torques; then the
mean over episodes (every episode weighs the same, whatever its length).
"""
import numpy as np

from .outputs import OUTPUT_NAMES


def errors(truth, prediction, scale):
    error = np.asarray(prediction) - np.asarray(truth)
    if error.ndim != 2 or error.shape[1] != 9 or not len(error) or not np.isfinite(error).all():
        raise ValueError("Invalid evaluation arrays")
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    return error_components(mae, rmse, scale, len(error))


def error_components(mae, rmse, scale, rows):
    """Format physical moments; shared by in-memory and streaming evaluation."""
    nrmse = rmse / scale
    return {
        'rows': rows,
        'per_output': {name: {'mae': float(mae[i]), 'rmse': float(rmse[i]),
                              'nrmse_train_std': float(nrmse[i]),
                              'unit': 'rad/s' if i < 2 else 'N*m'}
                       for i, name in enumerate(OUTPUT_NAMES)},
        'wheels_macro_mae_rad_s': float(mae[:2].mean()),
        'wheels_macro_rmse_rad_s': float(rmse[:2].mean()),
        'arm_macro_mae_nm': float(mae[2:].mean()),
        'arm_macro_rmse_nm': float(rmse[2:].mean()),
        'balanced_nrmse': float(.5 * nrmse[:2].mean() + .5 * nrmse[2:].mean()),
    }


def aggregate(episodes):
    """Equal weight per episode; primary metric also weights wheel/arm groups equally."""
    keys = ['wheels_macro_mae_rad_s', 'wheels_macro_rmse_rad_s',
            'arm_macro_mae_nm', 'arm_macro_rmse_nm', 'balanced_nrmse']
    values = {key: float(np.mean([e[key] for e in episodes])) for key in keys}
    per_output = {name: {**{key: float(np.mean([e['per_output'][name][key] for e in episodes]))
                                   for key in ['mae', 'rmse', 'nrmse_train_std']},
                         'unit': episodes[0]['per_output'][name]['unit']}
                  for name in OUTPUT_NAMES}
    return {'episodes': len(episodes), 'episode_macro': values,
            'per_output_episode_macro': per_output,
            'balanced_nrmse_episode_median': float(np.median([e['balanced_nrmse'] for e in episodes])),
            'balanced_nrmse_episode_p90': float(np.percentile([e['balanced_nrmse'] for e in episodes], 90))}
