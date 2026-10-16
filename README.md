# COMP0245 — Final Project (2026/27)

Group project for Machine Learning for Robotics, MSc Robotics and AI, UCL.
Goal: behaviour cloning of a mobile manipulator controller, learning arm torques
and wheel commands from the robot state and target.

## Project files

- `docs/COMP0245-Assignment-2026-2.pdf`: assignment slides.
- `final/robotnik_arm_task/`: lecturer's final assignment starter code and running instructions.
- `week1/` and `week2/`: lecturer's lab material, retained for reference.

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

The lecturer's Git history is retained. At setup, the Python source is unchanged
from upstream commit `cae4fb69386872ae73c601ce77e182aad7abe284`; only this README,
the root `.gitignore`, and the assignment PDF were added or updated. As the project
progresses, document the group's new or modified code, data collection, training,
evaluation, and reproduction commands here, as required by the assignment.

---

## Original upstream README

# LAB_SESSION_COMP_0245_2026_PUBLIC

this repository is used to release the lab sessions for COMP0245 machine learning for robotics for the year 2026-2027 of the master in robotics and ai

due to the block module structure the lab sessions are organised in weekly folders such as week1, which contains the material for the lab sessions of that week

each weekly folder contains a README with the instructions for the lab session and the code used during the session

the configs and models used during the labs are intentionally not part of this repository, refer to the instructions in each weekly README to obtain them