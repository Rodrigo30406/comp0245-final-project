"""Step 9: the obstacle task (B1 "fork in the road") built on top of the A-to-B pipeline.

Shared modules (used by the simulation side and the learning side):
  scenario.py  scenario definition, sampler with its own RNG stream, JSON I/O
  lidar.py     analytic 2D lidar against the scenario's cylinders (the student's added sensor)

No file outside this package is modified by step 9; existing modules are imported, never edited.
"""
