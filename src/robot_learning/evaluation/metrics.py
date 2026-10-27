"""External sampled task metrics, independent of expert phases and stop labels."""
import numpy as np


class TaskMetrics:
    def __init__(self, initial_distance, dt, horizon_steps, tolerances, dwell_s, failure_penalty):
        if not np.isfinite(initial_distance) or initial_distance < 0:
            raise ValueError('Invalid initial distance')
        if dt <= 0 or horizon_steps < 1 or dwell_s <= 0 or failure_penalty <= 0:
            raise ValueError('Invalid task metric parameters')
        self.dt = dt; self.horizon_steps = horizon_steps
        self.dwell_steps = int(np.ceil(dwell_s/dt-1e-10))
        self.penalty = failure_penalty
        self.tolerances = tolerances
        self.minimum = self.last = float(initial_distance); self.integral = 0.; self.steps = 0
        self.starts = {str(t): (0 if initial_distance <= t else None) for t in tolerances}
        self.arrival = {str(t): None for t in tolerances}
        self.longest = {str(t): 0 for t in tolerances}

    def update(self, distance):
        if not np.isfinite(distance) or distance < 0:
            raise FloatingPointError('Invalid EE distance')
        self.steps += 1
        self.integral += .5 * (self.last+distance)*self.dt
        self.last = float(distance); self.minimum = min(self.minimum, distance)
        for t in self.tolerances:
            key = str(t)
            if distance <= t:
                if self.starts[key] is None:
                    self.starts[key] = self.steps
                duration = self.steps-self.starts[key]
                self.longest[key] = max(self.longest[key],duration)
                if duration >= self.dwell_steps and self.arrival[key] is None:
                    self.arrival[key] = self.steps*self.dt
            else:
                self.starts[key] = None

    def result(self, valid):
        horizon = self.horizon_steps*self.dt
        # Failure is explicit and penalized; never present a short invalid run as
        # a favorable mean over a smaller time interval.
        return {'valid_full_horizon': bool(valid), 'executed_steps': self.steps,
                'elapsed_simulated_s': self.steps*self.dt, 'horizon_s': horizon,
                'last_valid_distance_m': self.last,
                'final_distance_m': self.last if valid else None,
                'minimum_distance_m': float(self.minimum),
                'mean_distance_m': self.integral/horizon if valid else None,
                'penalized_final_distance_m': self.last if valid else max(self.last,self.penalty),
                'penalized_mean_distance_m': (self.integral +
                   (0 if valid else max(0,horizon-self.steps*self.dt)*self.penalty))/horizon,
                'by_tolerance': {str(t): {
                    'success': bool(valid and self.arrival[str(t)] is not None),
                    'stable_arrival_completion_s': self.arrival[str(t)] if valid else None,
                    'final_within_tolerance': bool(valid and self.last <= t),
                    'longest_sampled_dwell_s': self.longest[str(t)]*self.dt,
                } for t in self.tolerances}}
