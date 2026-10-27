# Notes and pending experiments

Record of ideas discussed with Rodrigo. **Update: 23/10/2026.** These are proposals for testing; no results from these experiments have been obtained yet.

## 1. Pilot start point

- Sample targets uniformly within a Cartesian box.
- Initial proposed box: X and Y between −2 and 2 m; Z between 0.3 and 0.8 m. These limits are pending validation with the expert.
- Restart each episode in the same nominal robot pose: initial base position and orientation, and initial arm configuration remain constant.
- Keep the original target for input and for computing arrival error. The expert's auxiliary target limited to 0.85 m does not replace the task target.
- Record targets, initial conditions, seeds, results, and reasons for termination. Also preserve failed attempts.

This proposed box updates the initial sector/radius sampling approach from the original plan. Examples of radial extrapolation from the original plan must be redefined when the training box is fixed: a radius greater than 1.4 m no longer automatically implies being outside a box with X/Y between −2 and 2 m.

## 2. Rodrigo's idea: increase horizontal distance and number of points separately

**Question:** How does behavior change when expanding the horizontal space of targets, apart from increasing the number of training points?

The relevant distance is the initial horizontal separation between base and target:

$$d_{xy}=\sqrt{(x_g-x_{b,0})^2+(y_g-y_{b,0})^2}.$$

| Experiment | What changes | What is maintained |
|---|---|---|
| Number of data points | Number of episodes/targets in training | Box, distribution, initial pose, architecture, and evaluation protocol |
| Horizontal extent | X/Y limits of the box | Number of episodes, Z range, initial pose, architecture, and expert parameters |
| Post-interaction behavior | Box size × number of episodes | Rest of the protocol |

As a preliminary sweep, consider half-box sizes of 1, 2, 3, and 4 meters. These are not approved limits nor proven reachability; begin by evaluating with the expert. Maintain the same height or the same set of heights initially to isolate horizontal distance effects.

**Important control:** expanding a box with the same number of points reduces its sampling density and changes the navigation difficulty. Report both effects; complement with an experiment maintaining approximately constant density and with metrics grouped by distance. Compare models both in a common region and across the full region of each box, indicating that these two averages answer different questions.

Save success, final error relative to the original target, arrival time, navigation time, distance traveled, and percentage of timeouts. Always compare against the expert: an increase in navigation boundary failures does not alone indicate that the model requires more data or greater capacity.

## 3. Rodrigo's idea: randomize the initial position

**Question:** Does the behavior change when the robot starts at different initial positions within the same square?

Proposed sequence:

1. Pilot with fixed nominal pose and variable targets.
2. Evaluate the trained model from new initial positions: measure generalization without additional training.
3. Collect another dataset with uniformly random initial X/Y positions within the square and train a variant: measure the effect of teaching this variation.
4. Compare both policies against the same reserved scenarios, first with equal numbers of training episodes.
5. Later, vary the initial base orientation and arm posture in separate experiments.

In the first variation of position, maintain fixed base orientation, joint angles, and initial velocities. Verify a consistent reset with the floor and geometry. Keep common targets across comparisons and record relative distance and relative heading; the expert's strategy also depends on the absolute coordinates of the target, so do not assume translational invariance.

**Geometric consequence:** Starting fixed at the center of a 4 × 4 m box, the maximum horizontal distance to the target is approximately 2.83 m. If both start and target can be at opposite corners of the same box, the distance reaches approximately 5.66 m. Therefore, randomizing the start also increases possible distances. Compare by distance intervals and reserve an experiment with equidistant targets to separate this effect from that of changing absolute position.

## 4. What does the 25-second limit mean?

Verified in the current code:

- `BASE_NAV_TIMEOUT = 25.0` is passed to the base navigation controller.  
- `current_time` starts at zero and increases with each physical step. It is directly compared with `nav_timeout`; it is not reset upon movement or stopping.  
- It is **25 seconds of simulated time from the start of the episode for navigation**, counting rotations, forward motion, and final orientation. It is not a timer for being stationary or for lack of progress.  
- If navigation ends early, the arm begins reaching earlier.  
- If the time exceeds 25 s without reaching the target, the controller sets `done=True` equally. The arm then attempts reaching from where the base left off, and in subsequent calls, the base receives zero velocities. This does not imply that the desired position has been reached, nor does it conclude the entire episode.  
- The independent global limit is `ROBOENV_MAX_STEPS`, defaulting to 40,000. With a local `dt` of 0.001 s, this corresponds approximately to 40 seconds of simulated time **in total**, not an additional 40 seconds after navigation. Always verify the effective `dt`.  
- There is another stopping condition due to stagnation: during the reaching phase, it terminates if more than 3 s have passed since the last significant improvement in the EE error, with an improvement threshold of 0.002 m and a minimum temporal condition of over 8 s in the episode. This measures lack of progress toward the target, not robot immobility.  

To expand the time budget, first maintain the timeout and quantify its effect. A subsequent sweep of temporal budgeting is another experiment: do not silently increase it in certain cases and attribute all the difference to the data.  

Sources: [Episode clock](../src/robot_learning/simulation/episode.py), [Expert parameters](../experiment_specs/expert/default.json), [Base control and stopping criteria](../src/robot_learning/expert/controllers.py), [Overall plan](RESEARCH_PLAN.md).
