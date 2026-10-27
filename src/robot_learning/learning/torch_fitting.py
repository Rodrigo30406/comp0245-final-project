"""Optional PyTorch fitting; export the existing portable NumPy MLP policy."""
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from robot_learning.data.storage import read_json, sha256, write_json
from .dataset import load_episode
from .metrics import errors
from .model import MLPPolicy


def validate_torch_config(settings):
    if set(settings) != {'device', 'data_placement', 'validation_batch_size', 'deterministic'}:
        raise ValueError('Invalid PyTorch settings')
    if settings['device'] not in ('auto', 'cuda', 'cpu'):
        raise ValueError('PyTorch device must be cuda, cpu or auto')
    if settings['data_placement'] not in ('auto', 'device', 'stream'):
        raise ValueError('data_placement must be auto, device or stream')
    if type(settings['validation_batch_size']) is not int or settings['validation_batch_size'] < 1:
        raise ValueError('validation_batch_size must be a positive integer')
    if type(settings['deterministic']) is not bool:
        raise ValueError('deterministic must be boolean')


def build_network(hp, device, input_features=21):
    """Use the same NumPy Glorot/bias draws as the frozen sklearn regressor."""
    import torch
    widths = [input_features, *hp['hidden_layer_sizes'], 9]
    layers=[]
    for i,(fan_in,fan_out) in enumerate(zip(widths[:-1],widths[1:])):
        layers.append(torch.nn.Linear(fan_in,fan_out))
        if i < len(widths)-2:layers.append(torch.nn.ReLU())
    network = torch.nn.Sequential(*layers).to(device=device, dtype=torch.float32)
    rng = np.random.RandomState(hp['seed'])
    with torch.no_grad():
        for layer, fan_in, fan_out in zip(network[::2], widths[:-1], widths[1:]):
            bound = np.sqrt(6. / (fan_in + fan_out))
            weight = rng.uniform(-bound, bound, (fan_in, fan_out)).astype(np.float32)
            bias = rng.uniform(-bound, bound, fan_out).astype(np.float32)
            layer.weight.copy_(torch.from_numpy(weight.T.copy()).to(device))
            layer.bias.copy_(torch.from_numpy(bias).to(device))
    return network, rng


def export_policy(network, normalizers, window=None):
    return MLPPolicy(*normalizers,
        [layer.weight.detach().cpu().numpy().T.copy() for layer in network[::2]],
        [layer.bias.detach().cpu().numpy().copy() for layer in network[::2]], window=window)


def batch_objective(network, x, y, alpha):
    """Match sklearn's actual multioutput gradients and its reported loss.

    sklearn reports half mean squared error over outputs but its backpropagation
    sums output gradients. Keep both conventions explicit; regularize weights
    only, with alpha divided by the actual minibatch size (including the tail).
    """
    import torch
    squared = (network(x) - y).square()
    penalty = (.5 * alpha / len(x)) * sum(layer.weight.square().sum() for layer in network[::2])
    objective = .5 * squared.sum(dim=1).mean() + penalty
    reported_loss = .5 * squared.mean() + penalty
    return objective, reported_loss.detach()


def adam_step(optimizer, step):
    """Map sklearn's uncorrected-moment epsilon to PyTorch Adam's convention."""
    for group in optimizer.param_groups:
        group['eps'] = 1e-8 / math.sqrt(1. - group['betas'][1] ** step)
    optimizer.step()


def validation_cache(dataset, records, scaler, folder, window=None):
    folder.mkdir()
    rows = sum(r['rows'] for r in records)
    x = np.lib.format.open_memmap(folder/'x.npy', mode='w+', dtype='float32', shape=(rows, 21))
    y = np.lib.format.open_memmap(folder/'y.npy', mode='w+', dtype='float64', shape=(rows, 9))
    ranges = []; offset = 0
    for record in records:
        a, b, _ = load_episode(dataset, record, window=window); end = offset + len(a)
        x[offset:end] = scaler.transform(a).astype(np.float32); y[offset:end] = b
        ranges.append((offset, end)); offset = end
    x.flush(); y.flush()
    return x, y, ranges


def validation_score(network, vx, vy, ranges, normalizers, device, batch_size, resident=None, window=None):
    """GPU forward passes; float64 physical errors with equal episode weights."""
    import torch
    scores = []
    if window is not None:
        from .windows import encode_numpy, encode_torch, lags
        starts = np.array([lo for lo, hi in ranges], dtype=np.int64)
        tensor_starts = torch.from_numpy(starts).to(device)
        offsets = torch.from_numpy(lags(window)).to(device)
    network.eval()
    with torch.inference_mode():
        for lo, hi in ranges:
            predicted = np.empty((hi-lo, 9), dtype=np.float64)
            for start in range(lo, hi, batch_size):
                end = min(start+batch_size, hi)
                if window is None:
                    inputs = (resident[start:end] if resident is not None else
                              torch.from_numpy(vx[start:end]).to(device))
                elif resident is not None:
                    inputs = encode_torch(resident, torch.arange(start, end, device=device), tensor_starts, offsets)
                else:
                    inputs = torch.from_numpy(encode_numpy(vx, np.arange(start, end), starts, window)).to(device)
                value = network(inputs).cpu().numpy()
                predicted[start-lo:end-lo] = value * normalizers[3] + normalizers[2]
            scores.append(errors(vy[lo:hi], predicted, normalizers[3])['balanced_nrmse'])
    return float(np.mean(scores))


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def fit_torch(x, y, x_scaler, y_scaler, config, dataset, validation_records, output,
              *, mapped=False, checkpoint_context=None, resume_checkpoint=None, training_ranges=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    started = time.perf_counter(); hp = config['mlp']; settings = config['torch']
    validate_torch_config(settings)
    requested = settings['device']
    device = torch.device('cuda' if requested == 'auto' and torch.cuda.is_available() else
                          'cpu' if requested == 'auto' else requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable; no implicit CPU fallback')
    torch.manual_seed(hp['seed'])
    torch.use_deterministic_algorithms(settings['deterministic'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    normalizers = (x_scaler.mean_, x_scaler.scale_, y_scaler.mean_, y_scaler.scale_)
    window = config.get('observation_window')
    if window is not None:
        from .windows import encode_numpy, encode_torch, lags
        if (not training_ranges or training_ranges[0][0] != 0 or training_ranges[-1][1] != len(x) or
                any(a[1] != b[0] for a, b in zip(training_ranges, training_ranges[1:]))):
            raise ValueError('Window fitting requires complete contiguous training episode ranges')
        starts = np.array([lo for lo, hi in training_ranges], dtype=np.int64)
        tensor_starts = torch.from_numpy(starts).to(device)
        offsets = torch.from_numpy(lags(window)).to(device)
    if mapped:
        from .workspace import normalized_maps
        xn, yn = normalized_maps(x, y, x_scaler, y_scaler, output)
    else:
        xn = x_scaler.transform(x).astype(np.float32)
        yn = y_scaler.transform(y).astype(np.float32)
    vx, vy, ranges = validation_cache(dataset, validation_records, x_scaler, output/'torch_validation', window=window)
    network, rng = build_network(hp, device, input_features=21 if window is None else 18*window['snapshots']+3)
    optimizer = torch.optim.Adam(network.parameters(), lr=hp['learning_rate'],
        betas=(.9, .999), eps=1e-8, weight_decay=0., fused=device.type == 'cuda')
    gpu_info = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        free, total = torch.cuda.mem_get_info(device)
        gpu_info = {'name': torch.cuda.get_device_name(device), 'total_bytes': total,
                    'free_bytes_before_data': free, 'capability': list(torch.cuda.get_device_capability(device))}
        estimate = xn.nbytes + yn.nbytes + vx.nbytes + 16*len(xn) + 256*1024**2
        resident = settings['data_placement'] == 'device' or (
            settings['data_placement'] == 'auto' and estimate <= .75*free)
        if resident and estimate > .9*free:
            raise ValueError('Requested device-resident data exceeds the conservative free-VRAM budget; use auto/stream')
    else:
        resident = settings['data_placement'] != 'stream'
    tx = torch.from_numpy(xn).to(device) if resident else None
    ty = torch.from_numpy(yn).to(device) if resident else None
    tvx = torch.from_numpy(vx).to(device) if resident else None
    placement = 'device' if resident else 'stream'
    print(f'PyTorch {torch.__version__}; device={device}; data={placement}; GPU={gpu_info}', flush=True)
    context = {**(checkpoint_context or {}), 'backend': 'pytorch', 'torch_version': str(torch.__version__),
               'cuda_runtime': torch.version.cuda, 'device': str(device),
               'gpu_name': gpu_info['name'] if gpu_info else None}
    signature = json.loads(json.dumps(config)); signature.pop('name'); signature['mlp'].pop('max_epochs')
    history = []; best_score = math.inf; patience_score = math.inf; stale = 0
    best_epoch = 0; best_state = None; first_epoch = 1; steps = 0
    if resume_checkpoint is not None:
        path = Path(resume_checkpoint)
        header = read_json(path if path.suffix == '.json' else path.with_suffix('.json'))
        if path.suffix == '.json':path = path.parent/header['file']
        if (header.get('backend') != 'pytorch' or header['context'] != context or
                header['config_without_cap'] != signature or sha256(path) != header['sha256']):
            raise ValueError('PyTorch checkpoint integrity/configuration/data/backend mismatch')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        for actual, old in zip(normalizers, saved['normalizers']):
            np.testing.assert_array_equal(actual, old.numpy())
        network.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        state = saved['numpy_rng']; rng.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':torch.cuda.set_rng_state(saved['cuda_rng'], device)
        history = saved['history']; best_score = saved['best_score']; best_epoch = saved['best_epoch']
        patience_score = saved['patience_score']; stale = saved['stale']; steps = saved['steps']
        best_state = saved['best_model']; first_epoch = saved['epoch']+1
        write_json(output/'history.json', history)
    epoch = first_epoch-1
    for epoch in itertools.count(first_epoch):
        if stale >= hp['patience'] or (hp['max_epochs'] is not None and epoch > hp['max_epochs']):
            epoch -= 1; break
        epoch_start = time.perf_counter(); network.train()
        # One full deterministic permutation per epoch; preserve sklearn's RNG stream.
        order = rng.permutation(len(xn))
        indices = torch.from_numpy(order).to(device) if resident else None
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        for start in range(0, len(xn), hp['batch_size']):
            end = min(start+hp['batch_size'], len(xn))
            if resident:
                batch = indices[start:end]; bx = tx[batch]; by = ty[batch]
                if window is not None:
                    bx = encode_torch(tx, batch, tensor_starts, offsets)
            else:
                batch = order[start:end]
                features = xn[batch] if window is None else encode_numpy(xn, batch, starts, window)
                bx = torch.from_numpy(features).to(device); by = torch.from_numpy(yn[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            objective, reported = batch_objective(network, bx, by, hp['alpha'])
            objective.backward(); steps += 1; adam_step(optimizer, steps)
            loss_sum.add_(reported.double() * (end-start))
        loss = float((loss_sum / len(xn)).item())
        score = validation_score(network, vx, vy, ranges, normalizers, device,
                                 settings['validation_batch_size'], tvx, window=window)
        if not math.isfinite(loss) or not math.isfinite(score):
            raise ValueError('Nonfinite PyTorch loss/validation error')
        if score < best_score:
            best_score = score; best_epoch = epoch; best_state = cpu_tree(network.state_dict())
            export_policy(network, normalizers, window=window).save(output/'model.npz')
        if score < patience_score-hp['min_delta']:
            patience_score = score; stale = 0
        else:
            stale += 1
        history.append({'epoch': epoch, 'training_normalized_loss': loss,
                        'validation_episode_macro_balanced_nrmse': score,
                        'seconds': time.perf_counter()-epoch_start})
        write_json(output/'history.json', history)
        folder = output/'optimizer_checkpoints'; folder.mkdir(exist_ok=True)
        path = folder/f'epoch_{epoch:06d}.pt'; temp = path.with_suffix('.tmp')
        rs = rng.get_state()
        torch.save({'model': cpu_tree(network.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
            'normalizers': [torch.from_numpy(a.copy()) for a in normalizers],
            'numpy_rng': [rs[0], rs[1].tolist(), *rs[2:]], 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            'epoch': epoch, 'steps': steps, 'history': history, 'best_score': best_score,
            'best_epoch': best_epoch, 'patience_score': patience_score, 'stale': stale,
            'best_model': best_state}, temp)
        temp.replace(path)
        header = {'backend': 'pytorch', 'schema_version': 1, 'context': context,
                  'config_without_cap': signature, 'sha256': sha256(path), 'epoch': epoch, 'file': path.name}
        write_json(path.with_suffix('.json'), header)
        write_json(output/'optimizer_latest.json', {**header, 'file': str(path.relative_to(output))})
        print(f'Epoch {epoch}: loss={loss:.6f}, validation NRMSE={score:.6f}, best={best_score:.6f}', flush=True)
    network.load_state_dict(best_state)
    policy = export_policy(network, normalizers, window=window); policy.save(output/'model.npz')
    preview_x = xn[:128] if window is None else encode_numpy(xn, np.arange(min(128,len(xn))), starts, window)
    with torch.inference_mode():
        preview = network(torch.from_numpy(preview_x).to(device)).cpu().numpy()
    numpy_prediction = (policy.predict(x[:128]) if window is None else
        policy.predict_encoded(encode_numpy(x, np.arange(min(128,len(x))), starts, window)))
    torch_prediction = preview*normalizers[3]+normalizers[2]
    np.testing.assert_allclose((numpy_prediction-normalizers[2])/normalizers[3], preview,
                               rtol=1e-4, atol=2e-5)
    info = {'kind': 'mlp_regressor', 'backend': 'pytorch', 'training_device': str(device),
        'torch_version': str(torch.__version__), 'cuda_runtime': torch.version.cuda, 'gpu': gpu_info,
        'data_placement': placement, 'dtype': 'float32', 'mixed_precision': False, 'tf32': False,
        'deterministic_algorithms': settings['deterministic'],
        'hidden_layer_sizes': hp['hidden_layer_sizes'], 'activation': 'relu', 'output_activation': 'identity',
        'optimizer': 'Adam; sklearn-equivalent uncorrected-moment epsilon via adjusted PyTorch eps',
        'regression_parameters': sum(p.numel() for p in network.parameters()), 'weight_dtype': 'float32',
        'epochs_completed': epoch, 'best_epoch': best_epoch,
        'best_validation_episode_macro_balanced_nrmse': best_score,
        'stop_reason': 'patience' if stale >= hp['patience'] else 'max_epochs',
        'checkpoint_selection': 'minimum equal-episode validation balanced NRMSE',
        'training_updates': steps, 'training_examples_seen': epoch*len(xn),
        'training_loss': 'sklearn reported half output-mean MSE; gradients use half output-sum MSE; weight-only alpha/(2*batch_rows) L2',
        'initialization': 'Same NumPy RandomState Glorot uniform weights/bias draws and epoch permutation stream as sklearn',
        'backend_equivalence': 'Same architecture/data/objective conventions; different numeric kernels, not bit-identical training',
        'validation_for_stopping': 'GPU/CPU torch forward; float64 physical errors; equal episodes and 50/50 wheel/arm groups',
        'inference_backend': 'Existing NumPy MLPPolicy; simulator needs no torch/CUDA',
        'export_preview_max_physical_difference': float(np.max(np.abs(numpy_prediction-torch_prediction))),
        'resumed_from_epoch': first_epoch-1, 'epochs_executed_this_run': epoch-first_epoch+1,
        'optimizer_checkpoint': 'optimizer_latest.json; hash-verified PyTorch weights-only load, Adam/RNG/best/patience state',
        'cumulative_fit_validation_seconds': sum(h['seconds'] for h in history),
        'fit_and_transform_seconds': time.perf_counter()-started}
    if device.type == 'cuda':
        info['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        info['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
    del vx, vy, tvx
    shutil.rmtree(output/'torch_validation')
    return policy, info
