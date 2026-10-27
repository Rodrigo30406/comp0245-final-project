"""Causal observation stacks shared by offline fitting and live inference."""
import numpy as np


def validate_window(spec):
    if (set(spec) != {'snapshots', 'stride_steps', 'dt_s', 'padding'} or
            type(spec['snapshots']) is not int or spec['snapshots'] < 2 or
            type(spec['stride_steps']) is not int or spec['stride_steps'] < 1 or
            not isinstance(spec['dt_s'], (int, float)) or not np.isfinite(spec['dt_s']) or
            spec['dt_s'] <= 0 or spec['padding'] != 'repeat_initial_state'):
        raise ValueError('Invalid causal observation window')


def lags(spec):
    """Oldest to newest; the final state is always the current observation."""
    return np.arange(spec['snapshots']-1, -1, -1, dtype=np.int64)*spec['stride_steps']


def feature_names(spec, names):
    return [f'{name}_lag_{int(lag)}_steps' for lag in lags(spec) for name in names[:18]] + names[18:]


def encode_numpy(base, indices, starts, spec):
    """Gather one batch, clamping each lag at its own episode's first row.

    base can contain raw or individually normalized 21D observations. starts
    contains the first row of each episode, in the concatenated observation bank.
    No overlapping window array is retained, and no goal history is repeated.
    """
    indices = np.asarray(indices, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    origin = starts[np.searchsorted(starts, indices, side='right')-1]
    past = np.maximum(indices[:, None]-lags(spec), origin[:, None])
    return np.concatenate([base[past, :18].reshape(len(indices), -1), base[indices, 18:]], axis=1)


def encode_torch(base, indices, starts, offsets):
    """Equivalent on-device gather; only the current minibatch is expanded."""
    import torch
    origin = starts[torch.searchsorted(starts, indices, right=True)-1]
    past = torch.maximum(indices[:, None]-offsets, origin[:, None])
    return torch.cat([base[past, :18].reshape(len(indices), -1), base[indices, 18:]], dim=1)


def validate_episode_sampling(data, metadata, spec):
    """Fail closed instead of treating sparse logged rows as physical steps."""
    if 'step' not in data or 'time' not in data or 'dt' not in metadata:
        raise ValueError('Temporal training requires recorded steps, times and dt')
    step, time = data['step'], data['time']
    if (not np.isclose(metadata['dt'], spec['dt_s'], rtol=0, atol=1e-12) or
            not np.array_equal(step, np.arange(len(step))) or
            not np.allclose(time, step*spec['dt_s'], rtol=0, atol=1e-8)):
        raise ValueError('Temporal training requires dense observations from step zero at the configured dt')


class OnlineHistory:
    """Bounded measured-state ring; one append per physical policy call."""
    def __init__(self, spec):
        self.spec = spec
        self.capacity = int(lags(spec)[0])+1
        self.states = np.empty((self.capacity, 18), dtype=np.float64)
        self.count = 0

    def append(self, observation):
        x = np.asarray(observation, dtype=np.float64)
        if x.shape != (21,) or not np.isfinite(x).all():
            raise ValueError('Expected finite current 21D observation')
        if self.count == 0:
            self.states[:] = x[:18]
        self.states[self.count % self.capacity] = x[:18]
        indices = np.maximum(self.count-lags(self.spec), 0) % self.capacity
        self.count += 1
        return np.concatenate([self.states[indices].reshape(-1), x[18:]])
