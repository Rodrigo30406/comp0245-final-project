"""Nine independent controls and portable normalized linear inference."""
from pathlib import Path
import numpy as np

INPUT_NAMES = ([f"base_position_{a}" for a in "xyz"] + ["base_bearing"] +
               [f"arm_q_{i}" for i in range(1, 8)] +
               [f"arm_qd_{i}" for i in range(1, 8)] +
               [f"target_{a}" for a in "xyz"])
OUTPUT_NAMES = ["wheel_right", "wheel_left"] + [f"arm_torque_{i}" for i in range(1, 8)]
ACTION_NAMES = ["wheel_front_right", "wheel_front_left", "wheel_back_left", "wheel_back_right"] + OUTPUT_NAMES[2:]


def independent_actions(actions):
    """Require the expert's exact same-side wheel identity; never average it away."""
    a = np.asarray(actions)
    if a.ndim != 2 or a.shape[1] != 11 or not np.isfinite(a).all():
        raise ValueError("Expected finite (N, 11) expert actions")
    if not np.array_equal(a[:, 0], a[:, 3]) or not np.array_equal(a[:, 1], a[:, 2]):
        raise ValueError("Expert wheel commands do not have identical same-side pairs")
    return a[:, [0, 1, *range(4, 11)]]


def expand_actions(actions):
    a = np.asarray(actions)
    if a.shape[-1] != 9 or not np.isfinite(a).all():
        raise ValueError("Expected finite nine-output controls")
    return a[..., [0, 1, 1, 0, *range(2, 9)]]


class LinearPolicy:
    """Inference in physical units; no clipping or hidden controller adjustments."""
    def __init__(self, x_mean, x_scale, y_mean, y_scale, coef, intercept):
        self.x_mean = np.asarray(x_mean)
        self.x_scale = np.asarray(x_scale)
        self.y_mean = np.asarray(y_mean)
        self.y_scale = np.asarray(y_scale)
        self.coef = np.asarray(coef)
        self.intercept = np.asarray(intercept)
        if (self.x_mean.shape != (21,) or self.x_scale.shape != (21,) or
            self.y_mean.shape != (9,) or self.y_scale.shape != (9,) or
            self.coef.shape != (9, 21) or self.intercept.shape != (9,)):
            raise ValueError("Invalid linear policy shapes")
        if not all(np.isfinite(v).all() for v in self.arrays().values()):
            raise ValueError("Non-finite model parameters")
        if np.any(self.x_scale <= 0) or np.any(self.y_scale <= 0):
            raise ValueError("Scales must be positive")

    def arrays(self):
        return {k: getattr(self, k) for k in
                ("x_mean", "x_scale", "y_mean", "y_scale", "coef", "intercept")}

    def predict(self, observation):
        x = np.asarray(observation, dtype=np.float64)
        if x.shape[-1] != 21 or not np.isfinite(x).all():
            raise ValueError("Expected finite observations ending in 21 features")
        return (((x - self.x_mean) / self.x_scale) @ self.coef.T + self.intercept) * self.y_scale + self.y_mean

    def predict_action(self, observation):
        return expand_actions(self.predict(observation))

    def save(self, path):
        np.savez_compressed(Path(path), **self.arrays())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{k: data[k] for k in data.files})


class MLPPolicy:
    """Configurable ReLU hidden layers, linear output, portable NumPy inference."""
    def __init__(self, x_mean, x_scale, y_mean, y_scale, weights, biases, window=None):
        from .windows import validate_window
        if window is not None:
            validate_window(window)
        self.window = window
        self._history = None
        self.x_mean = np.asarray(x_mean)
        self.x_scale = np.asarray(x_scale)
        self.y_mean = np.asarray(y_mean)
        self.y_scale = np.asarray(y_scale)
        self.weights = [np.asarray(w) for w in weights]
        self.biases = [np.asarray(b) for b in biases]
        if any(v.shape != (n,) for v, n in [(self.x_mean, 21), (self.x_scale, 21),
                                           (self.y_mean, 9), (self.y_scale, 9)]):
            raise ValueError('Invalid MLP normalization shapes')
        if len(self.weights) < 2 or len(self.biases) != len(self.weights):
            raise ValueError('MLP requires at least one hidden layer and matched weights/biases')
        width = self.input_features
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],):
                raise ValueError('Invalid MLP layer shapes')
            width = w.shape[1]
            if width < 1 or (i == len(self.weights)-1 and width != 9):
                raise ValueError('Invalid MLP output width')
        if not all(np.isfinite(v).all() for v in self.arrays().values()):
            raise ValueError('Non-finite MLP parameters')
        if np.any(self.x_scale <= 0) or np.any(self.y_scale <= 0):
            raise ValueError('Scales must be positive')

    def arrays(self):
        result = {k: getattr(self, k) for k in ('x_mean', 'x_scale', 'y_mean', 'y_scale')}
        result.update({f'weight_{i}': w for i, w in enumerate(self.weights)})
        result.update({f'bias_{i}': b for i, b in enumerate(self.biases)})
        if self.window is not None:
            result['window_config'] = np.array([self.window['snapshots'], self.window['stride_steps'], self.window['dt_s']])
        return result

    @property
    def input_features(self):
        return 21 if self.window is None else 18*self.window['snapshots']+3

    def predict_encoded(self, features):
        """Raw encoded feature batch; has no effect on the live history buffer."""
        x = np.asarray(features, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != self.input_features or not np.isfinite(x).all():
            raise ValueError('Invalid encoded MLP observations')
        mean, scale = self.x_mean, self.x_scale
        if self.window is not None:
            mean = np.r_[np.tile(mean[:18], self.window['snapshots']), mean[18:]]
            scale = np.r_[np.tile(scale[:18], self.window['snapshots']), scale[18:]]
        hidden = ((x-mean)/scale).astype(self.weights[0].dtype)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            hidden = hidden @ w + b
            if i < len(self.weights)-1:
                np.maximum(hidden, 0, out=hidden)
        return hidden*self.y_scale+self.y_mean

    def predict(self, observation):
        x = np.asarray(observation, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != 21 or not np.isfinite(x).all():
            raise ValueError('Expected finite observations ending in 21 features')
        single = x.ndim == 1
        x = np.atleast_2d(x)
        result = np.empty((len(x), 9), dtype=np.float64)
        # Each batch belongs to one episode; history never crosses predict calls.
        for start in range(0, len(x), 4096):
            features = x[start:start+4096]
            if self.window is not None:
                from .windows import encode_numpy
                features = encode_numpy(x, np.arange(start, min(start+4096, len(x))), [0], self.window)
            result[start:start+4096] = self.predict_encoded(features)
        return result[0] if single else result

    def predict_action(self, observation):
        if self.window is not None and np.asarray(observation).ndim == 1:
            from .windows import OnlineHistory
            if self._history is None:
                self._history = OnlineHistory(self.window)
            return expand_actions(self.predict_encoded(self._history.append(observation)))
        return expand_actions(self.predict(observation))

    def reset(self, dt=None):
        if self.window is not None and dt is not None and not np.isclose(dt, self.window['dt_s'], rtol=0, atol=1e-12):
            raise ValueError('Policy window dt differs from simulator dt')
        self._history = None

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            count=sum(key.startswith('weight_') for key in data.files)
            expected={f'{prefix}_{i}' for prefix in ['weight','bias'] for i in range(count)}
            actual={key for key in data.files if key.startswith(('weight_','bias_'))}
            if actual != expected:
                raise ValueError('MLP layer indices must be contiguous with matching biases')
            spec = None
            if 'window_config' in data:
                n, stride, dt = data['window_config']
                if n != int(n) or stride != int(stride):
                    raise ValueError('Invalid serialized temporal configuration')
                spec = dict(snapshots=int(n), stride_steps=int(stride), dt_s=float(dt), padding='repeat_initial_state')
            return cls(**{k: data[k] for k in ('x_mean', 'x_scale', 'y_mean', 'y_scale')},
                       weights=[data[f'weight_{i}'] for i in range(count)],
                       biases=[data[f'bias_{i}'] for i in range(count)], window=spec)
