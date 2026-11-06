"""Post-hoc, descriptive (written 2026-11-03 after G was stopped by its frozen G5 rule; does not change any verdict).

Deployment variants of a chunk model, to separate "the chunk model" from "ACT-style temporal ensembling":
  j0      -- execute only segment j = 0 of the newest query (no ensembling)
  newest  -- ensembling with the weights reversed (newest prediction weighted most, same m = 0.11)
"""
import numpy as np

from policies.bc_policy import ChunkPolicy, wheel_speeds


class J0Policy(ChunkPolicy):
    def act(self, obs):
        y = np.asarray(self.predict(obs)).reshape(self.chunk_k, 9)[0]
        return wheel_speeds(y[0], y[1], self.base_params), y[2:9]


class NewestPolicy(ChunkPolicy):
    def ensemble(self, prediction):
        t = self._step
        self._chunks[t % self._buffer_size] = np.asarray(prediction).reshape(self.chunk_k, 9)
        js = np.arange(0, min(self.chunk_k - 1, t // self.chunk_delta) + 1)   # i = 0 is the newest (j = 0)
        weights = np.exp(-self.ensemble_m * np.arange(len(js)))
        values = self._chunks[(t - js * self.chunk_delta) % self._buffer_size, js]
        self._step += 1
        return np.sum(values * weights[:, None], axis=0) / weights.sum()


def make_policy(arg):
    """arg: '<variant>:<model path>'"""
    variant, path = arg.split(":", 1)
    return {"j0": J0Policy, "newest": NewestPolicy}[variant](path)
