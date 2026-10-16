# COMP0245 — Final Project (2026/27)

Group project for Machine Learning for Robotics, MSc Robotics and AI, UCL.
Goal: behaviour cloning of a mobile manipulator controller, learning arm torques
and wheel commands from the robot state and target.

## Project files

- `docs/COMP0245-Assignment-2026-2.pdf`: assignment slides.
- [Research plan (Spanish)](docs/PLAN_DE_INVESTIGACION.md): methodology, data design, model comparisons, evaluation protocol, and bibliography.
- `final/robotnik_arm_task/`: lecturer's final assignment starter code and running instructions.

## Run the starter simulation

Follow `final/robotnik_arm_task/README.md`: use your RoboEnv environment and make
its `models/` and `configs/` folders accessible inside `final/robotnik_arm_task/`.
Then run from that directory:

```bash
python robotnik_arm_task.py
```

Expected output: a PyBullet simulation of the robot reaching a target, followed
by plots. Simulator assets, generated datasets, and checkpoints are excluded
from Git. The simulation has not been validated during this repository setup.

## Working as a group

Clone the private group repository after the owner grants you access:

```bash
git clone https://github.com/Rodrigo30406/comp0245-final-project.git
cd comp0245-final-project
git switch -c feature/your-task
```

Commit and push your branch, then open a pull request into `main` for review.
Keep large datasets and checkpoints outside Git and document how to obtain them.
Each clone can add the lecturer's repository to receive future updates:

```bash
git remote add upstream https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC.git
git fetch upstream
```

Review upstream changes before merging them into the group's work.

## Submission and attribution

The slides specify a report of at most 6 pages using the IEEE LaTeX template,
plus a contribution statement of at most 1 page. The stated submission deadline
is 6 November 2026, with a required in-person demonstration during 30 October–6 November.

Starter code comes from:
https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC

The lecturer's Git history is retained. The final assignment Python source is
unchanged from upstream commit `cae4fb69386872ae73c601ce77e182aad7abe284`.
This project adds the assignment slides, research plan, root `.gitignore`, and
project documentation. The independent `week1/` and `week2/` lab exercises were
removed to keep this repository focused on the final assignment; they remain
available in upstream and Git history. The final assignment depends on RoboEnv's
`simulation_and_control` library, not on those lab folders.

As the project progresses, document the group's new or modified code, data
collection, training, evaluation, and reproduction commands here, as required by
the assignment.
