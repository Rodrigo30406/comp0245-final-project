"""Physical per-output errors and episode/group-balanced normalized errors."""
import csv
import numpy as np
from .model import OUTPUT_NAMES


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


class ErrorMoments:
    """Pool transition errors without retaining the complete prediction matrix."""
    def __init__(self):
        self.rows=0;self.absolute=np.zeros(9);self.squared=np.zeros(9)

    def add(self,truth,prediction):
        e=np.asarray(prediction)-np.asarray(truth)
        if e.ndim!=2 or e.shape[1]!=9 or not np.isfinite(e).all():
            raise ValueError('Invalid streaming errors')
        self.rows+=len(e);self.absolute+=np.sum(np.abs(e),axis=0)
        self.squared+=np.einsum('ij,ij->j',e,e)

    def result(self,scale):
        if not self.rows:raise ValueError('Empty error accumulator')
        return error_components(self.absolute/self.rows,np.sqrt(self.squared/self.rows),scale,self.rows)


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


def write_channel_report(path, metrics):
    """Readable per-channel report; preserve both episode and transition weighting."""
    columns = ['split', 'aggregation', 'output', 'unit', 'model_mae', 'model_rmse',
               'model_nrmse_train_std', 'mean_baseline_mae', 'mean_baseline_rmse',
               'mean_baseline_nrmse_train_std']
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for split, result in metrics.items():
            macro = {method: aggregate([episode[method] for episode in result['per_episode']])
                     ['per_output_episode_macro'] for method in ['model', 'mean_baseline']}
            for aggregation in ['episode_macro', 'pooled_transitions']:
                source = macro if aggregation == 'episode_macro' else {
                    method: result['pooled_transitions'][method]['per_output']
                    for method in ['model', 'mean_baseline']}
                for name in OUTPUT_NAMES:
                    row = {'split': split, 'aggregation': aggregation, 'output': name,
                           'unit': source['model'][name]['unit']}
                    for method in ['model', 'mean_baseline']:
                        row.update({f'{method}_{key}': source[method][name][key]
                                    for key in ['mae', 'rmse', 'nrmse_train_std']})
                    writer.writerow(row)
