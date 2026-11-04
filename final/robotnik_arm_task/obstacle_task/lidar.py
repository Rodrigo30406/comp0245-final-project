"""Analytic 2D lidar of the obstacle task (step 9): the student's added sensor.

A planar scanner at the base centre with N_BEAMS beams spread uniformly over 360 deg in the base frame
(beam 0 points along the base's forward axis, angles increase counter-clockwise), returning the distance to the
nearest obstacle surface, clipped to MAX_RANGE. The scenario's obstacles are vertical cylinders taller than the
scan plane, so the 2D ray-circle intersection is exact. The robot's own body is ignored (an idealised sensor;
pybullet rayTestBatch would also hit the chassis in ~2% of beams).

The grounding check (2026-11-03, scratchpad bench_sensors.py) found this caster within 0.1 mm of pybullet
rayTestBatch on 32/64-beam scans at 0.037 ms per 64-beam scan, so it is used both online (the harness puts the
scan in obs["lidar"]) and offline (training features recomputed from the logged base pose and the stored layout).
"""
import numpy as np

N_BEAMS = 64
MAX_RANGE = 4.0       # m
SCAN_HEIGHT = 0.35    # m, nominal scan-plane height (documentation only: every obstacle is taller than this)


def beam_angles(n_beams=N_BEAMS):
    """Beam directions in the base frame (rad): 0 = forward, counter-clockwise."""
    return 2.0 * np.pi * np.arange(n_beams) / n_beams


def scan_batch(base_xy, yaw, obstacles, n_beams=N_BEAMS, max_range=MAX_RANGE):
    """Ranges (N, n_beams) for N base poses. base_xy (N, 2), yaw (N,), obstacles [{"x", "y", "r", ...}]."""
    base_xy = np.atleast_2d(np.asarray(base_xy, dtype=float))
    yaw = np.atleast_1d(np.asarray(yaw, dtype=float))
    ang = yaw[:, None] + beam_angles(n_beams)[None, :]
    ux, uy = np.cos(ang), np.sin(ang)
    ranges = np.full(ang.shape, float(max_range))
    for ob in obstacles:
        cx = float(ob["x"]) - base_xy[:, 0]
        cy = float(ob["y"]) - base_xy[:, 1]
        c2 = cx * cx + cy * cy - float(ob["r"]) ** 2           # > 0: scanner outside the circle
        b = ux * cx[:, None] + uy * cy[:, None]                  # ray parameter of the closest approach
        disc = b * b - c2[:, None]
        near = b - np.sqrt(np.maximum(disc, 0.0))                # first intersection along the ray
        hit = (disc >= 0.0) & (near >= 0.0)
        t = np.where(hit, near, np.inf)
        t = np.where(c2[:, None] < 0.0, 0.0, t)                  # inside an obstacle: range 0
        ranges = np.minimum(ranges, t)
    return ranges


def scan(base_xy, yaw, obstacles, n_beams=N_BEAMS, max_range=MAX_RANGE):
    """Ranges (n_beams,) for one base pose."""
    return scan_batch(np.asarray(base_xy, dtype=float)[None, :2], np.array([float(yaw)]), obstacles,
                      n_beams, max_range)[0]
