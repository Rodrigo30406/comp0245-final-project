# COMP0245 final project: experimental line (Yudi Wang)

Behaviour cloning of the course's Robotnik + Kinova expert (A-to-B reaching), and an obstacle extension. This folder
is the *experimental line* of the group report: the diagnosis (experiments A, B), the triggering copies (C3), the
negative results (D, E, F, G, P, PC), the obstacle task (step 9), the cross-evaluation with the final recipe (step 10)
and the controls R, S, W (step 11). The *final recipe* of the report (Rodrigo Jesus Rodriguez Montes) is on the team
branch `rodrigo`; the multi-obstacle courses and mazes (Akshat Rana) are on `obstacle-akshat` / `obstacla-language`.

## What is from the course template, what is modified, what is new

| status | files |
|---|---|
| course template, **unchanged** | `robotnik_arm_task.py`, `robotnik_arm_control.py` (the expert: base state machine + arm controller), `configs/` (from RoboEnv); `models/` (robot descriptions from RoboEnv, not in git) |
| course template, **modified** | `README.md` (this overview added on top; the course text is kept below) |
| **written from scratch** | data: `collect_data.py` (DART-style held noise, per-episode `.npz`), `analyze_dataset.py`, `analyze_index.py`, `DATA_COLLECTION.md`; evaluation: `evaluate_policy.py` (closed-loop harness, protocol v1.1, verified step by step against the recordings), `analyze_rollouts.py` (failure taxonomy, two-level bootstrap, Welch), `EVALUATION.md`; learning: `bc_features.py`, `train_policy.py`, `train_step4e.py`, `dagger_relabel.py`, `policies/` (`bc_policy.py`, `hybrid.py`, `pd_layer.py`, `split_policy.py`, `rodrigo_policy.py` = adapter for the final recipe's weights); obstacle task: `obstacle_task/`; per-experiment scripts and read-outs: `results/step*/`; tests: `tests/`; lab notebook with every pre-registration and result: `report.txt`; experiment plans: `STEP*_PLAN_PROMPT.md` |

The only change to the expert's behaviour is the base goal: the standoff pose lies on the line from the robot's start
to the target instead of from the world origin (`evaluate_policy.expert_base_goal`, used by `collect_data.py`); it is
implemented in our code, and the two template files are byte-identical to the course repository.

## Running it
Environment: the course's RoboEnv pixi environment
(`pixi run --manifest-path <RoboEnv>/pixi.toml python ...`), run from this folder; `data/` and `models/` must be
present (models from RoboEnv; data regenerated with the commands in `DATA_COLLECTION.md` and `EVALUATION.md`:
train seed 1, pilot 1000, DEV 200, HARD 201, CONFIRM 202, test 100 (sealed, never opened); obstacle sets with
`results/step9/run_step9_sets.sh`).

| step | experiments | scripts / outputs |
|---|---|---|
| 1-2 | protocol, data analysis | `EVALUATION.md`, `evaluate_policy.py`, `analyze_dataset.py`, `results/data_analysis/` |
| 3 | A (wheel speed), B (base/arm swap), C3 (triggering copies), D (EE features), E (learning curve) | `results/step3/run_exp{A,B,C3,D}.sh`, `run_step4E.sh` |
| 5 | F0 tolerance curve, F (one DAgger round) | `results/step5/run_expF.sh` |
| 6-7 | G (chunking), P (PD output layer), PC (P confirmation) | `results/step6/run_exp{G,P}.sh`, `results/step7/run_expPC.sh` |
| 9 | obstacle task: L0, L1, L1-K0, L2; H-MM, H-FACT, H-COPY; dev2, confirm2, test2 (opened once) | `obstacle_task/`, `results/step9/` |
| 10 | cross-evaluation of the final recipe in this harness | `results/step10_cross/` (README there) |
| 11 | controls R (placebo row), S (two networks), W (capacity) | `results/step11/` (README there) |

Main models: C3 = `trained/ego-novw-trig_s{0..4}.npz` (DEV seeds) and `_s{100..104}` (CONFIRM seeds); obstacle L2 =
`trained/step9/L1arm_s*`, `L2cls_s*`, `L2base_s*`. Not in git (regenerable): `data/`, `models/`, feature caches,
1 kHz trajectories, per-episode and console logs.

---

# Course instructions (original README)

1) Make sure the folders "/models", "/configs" from RoboEnv are accessible (easiest is to copy them to this folder)
2) Use your RoboEnv environent (in whichever way you have been running the labs: pixi, uv, conda, etc)
3) Run robotnik_arm_task.py

Expected behaviour: pybullet 3D environment with 7dof arm mounted on a 4-wheeled platform. The robot moves to reach a target (red dot) with its end-effector. The simulation terminates when the target is reached, and plots of the robot behaviour are shown.

Key variables in robotnik_arm_task.py:

 - TARGET_EE_POS: reach target point in cartesian coordinates (x,y,z), represented by a red dot.

 - base_pos:     wheeled base position (x,y,z)
 - base_bearing: wheeled base bearing (angle in radians)
 - q_mes:        arm joint angles (radians)
 - qd_mes:       arm joint angle velocities (rad/s)

 - wheel_cmds:     angular velocity command for each wheel (rad/s)
 - torque_arm:     command torques for each arm joint
 - torque_fingers: torque for each finger joint of the end-effector grasper. In this work, they should just remain fixed.
 
 Assessment Goal:
 - ML model that produces (wheel_cmds, torque_arm) from measurements (base_pos,base_bearing,q_mes,qd_mes)

Step 5: tolerance curves and one round of DAgger (experiment F)
------------------------------------------------------------

The approved protocol is `STEP5_DAGGER_PLAN_PROMPT.md`. Its section 3 and the
approved F1 amendment 1 are frozen in `report.txt` and
`results/step5/prereg_F_frozen.txt`. Every stage checks their hashes.
Approved amendment 2 changes only descriptive F0 and its consistency rule.
The current plan SHA256 is
`554caf7ffea7364ac870b08eb85bbca80d06503d241a68d63e3e330066c8b43b`;
removing that paragraph reproduces the original frozen plan. The original
freeze records remain unchanged; the frozen section hash remains
`753a14e699d9c82421bf2c11589393f1a39f43bfe0b1dc66750b61744db6c06e`.
`amendment2_baseline.json` and `implementation_change2.json` record the change.

New code:

- `dagger_relabel.py`: sequential shadow-expert replay of full 1 kHz trajectories.
- `results/step5/tolerance_curve.py`: F0 curves from existing DEV, HARD and
  CONFIRM original rollout rows. DEV adds a separate C3 stalled replacement
  sensitivity curve (60.8% at 10 mm); HARD/CONFIRM use original rows only.
- `results/step5/build_dagger_cache.py`: F3 features, shadow labels and the
  lossless `results/step3/cache/dagger1/` side cache; descriptive `F_data.txt`.
- `results/step5/step5_pipeline.py`, `run_expF.sh`, `f_common.py`: resumable
  selection, pilot, simulations, training, integrity checks and sequential gates.
- `results/step5/F_mechanisms.py`: the frozen success criteria, M1-M4,
  descriptive secondary metrics with Holm correction and full comparisons.
- `tests/test_extra_rows.py`, `tests/test_f_gates.py`: training integration and
  sequential gate tests.

Modified code: `train_policy.py` adds the `ego-novw-trig-dag1` model and optional
`--extra-rows` for `train` and `train-many`. Extra rows are appended after the
trigger copies, with the original stride-9 standardisers and unchanged expert
validation. `tests/test_dagger_relabel.py` implements amendment 1, preserving
the first verification outputs. `report.txt` contains the frozen registration;
this README documents the execution.

Run from this directory in RoboEnv (all Python children inherit MPLCONFIGDIR):

```bash
nohup bash results/step5/run_expF.sh all >> results/step5/chainF.log 2>&1 < /dev/null &
step5_chain_pid=$!
wait "$step5_chain_pid"
```

The shell waits to retain the execution session. In another session, check
`results/step5/chainF.log` after `sleep 600`. Completed stages and complete runs
are verified and skipped. An interrupted run directory without `summary.txt`
is deleted before rerunning. `WORKERS=22` is the simulation default; `FJOBS=6`
is the training default, subject to the available-memory check. Training and
simulation do not overlap.

DEV uses ten seeds; HARD and CONFIRM each use five. HARD opens only after the
DEV success and risk gates pass; CONFIRM opens once after HARD passes.
`F_mechanisms.py --allow-confirm` is used only in that gated stage. A failed
gate completes the experiment with KEEP C3 and leaves later sets unopened.
F5 reruns C3 HARD/CONFIRM stalled episodes with `--no-early-stop` after each
set's gate passes, before its dag1 evaluations. F0 never requires those reruns.
`data/test` remains sealed.

Outputs: F1 verification JSON/logs; `tolerance_curve.{txt,json}` and
`fig_tolerance.{png,pdf}`; pilot diagnostics; `F2_selection.json`; 400 training
rollouts; the DAgger cache/manifest; `F_data.{txt,json}`;
`backward_compatibility.json`; 15 `trained/ego-novw-trig-dag1_s*.npz` models;
`model_checks.json`; gated evaluation runs; `F_MM_compare_<set>.{txt,json}`;
`F_mechanisms.{txt,json}`; stage timestamps and `CODEX_REPORT_F.md`. Errors
produce `chainF_blocked.json` with a resume command.

Execution status (2026-11-02, after AF_UNIX was enabled): frozen hashes and all
107 implementation/input hashes passed. The full suite passed (175 tests,
2 existing optional slow tests skipped), and the C3 seed 0 retraining arrays
are bit-identical to the original model (`backward_compatibility.json`).

Current execution status (amendment 2): the chain was found terminated at F2 at 04:14:25 UTC (exit code 143; exact termination time and cause unknown), and documented at 04:15:28 UTC. F0 and the pilot passed; F2 completed 240/400 episodes. Seed 3 has 34 partial rows and no summary, so resume will delete and rerun that incomplete directory. F3-F5 and F_mechanisms have not started; HARD/CONFIRM remain unopened. The exact error and resume commands are recorded in `results/step5/chainF_blocked.json` and the Chinese report. Later stages have not been run past this blocker.

Full amendment-2 tests: 177 tests passed, with 2 existing optional slow tests skipped. Final integrity checks cover 109 implementation/input files and 380 existing result files. See [the Chinese audit report](results/step5/CODEX_REPORT_F.md), `resume_final_integrity_amendment2.json`, and `chainF.log`; earlier session history is preserved.

第 6 步：实验 G（action chunking）与 P（关节参考输出）
-------------------------------------------------------

方案见 `STEP6_PLAN_PROMPT.md`。第一次执行因 P2(b) 失败按规则停止，证据保留；
本轮依据冻结前的“修订 1”继续实现与验证。最终测试、向后兼容证据、冻结、短链冒烟的
实际状态和时间以中文报告 `results/step6/REPORT_GP.md` 为准。本说明描述代码接口，
不将尚未运行的正式训练、pilot、DEV、HARD、CONFIRM 写成实验结果。

新增代码：

- `policies/pd_layer.py`：复用专家同一个 `ArmControllerParams`、PD 律与 Pinocchio
  重力补偿。按修订 1，固定层直接使用预测的 `e = q_des - q` 和 `qd_des`，不再对参考做
  lead 或关节限位截断，仅保留最终 `±max_torque` 力矩截断；专家 reach 分支的参考截断
  已包含在录制标签中。
- `results/step6/gp_training.py`：90 维动作块与 16 维关节参考的侧缓存、manifest、
  训练、峰值内存记录和离线诊断。G 早停使用全部 1 kHz 验证行的第 0 段；P 使用
  机械臂各维 `1/sqrt(2)` 权重，并以部署使用的固定层计算离线力矩。
- `gp_common.py`、`gp_mechanisms.py`、`G_mechanisms.py`、`P_mechanisms.py`
  （均在 `results/step6/`）：冻结核查、C3 stalled 对照替换、两层 bootstrap/Welch、
  Holm 描述量和 §6 的逐级门槛。
- `step6_pipeline.py`、`run_expG.sh`、`run_expP.sh`：缓存、pilot、G 的校准模型复用、
  训练、meta 检查、离线分析、DEV/HARD/CONFIRM 和图表的续跑链；`freeze_gp.py`
  写入并记录首次冻结。
- `gp_figures.py`：G pilot 曲线，DEV 的 C3/G/P success、失败类和 policy_ms 表，
  复用第 5 步定义的容差曲线，P 离线力矩 RMSE/参考 nRMSE 表，G reach 力矩平滑度。
  平滑度从共同 harness 检测的 `arm_extend_time` 开始，统计间隔 10 ms 的相邻存储点；
  缺失对应轨迹时拒绝静默替代。图表只读已完成结果，不重建评估参考或读取封存数据。
- `gp_smoke.py`：明确标注的软件合成 rollout/轨迹，用真实统计脚本检查门槛与自动停止。
  合成结果和少量真实闭环均隔离在 `results/step6/smoke/`，不参与正式判定。
- `tests/test_chunk.py`、`tests/test_pd_layer.py`、`tests/test_gp_figures.py` 及流水线测试：
  G4、修订后的 P2、描述性指标和门槛/续跑检查；以报告中的最终 failfast 运行记录为验收证据。

共享文件只添加默认关闭的入口：`train_policy.py` 登记 `ego-novw-trig-chunk`（G）、
`ego-novw-trig-pd`（P），新增 `--chunk-k`、`--chunk-delta`、`--side-cache` 和侧缓存
命令；旧模型仍进入原训练路径。`bc_features.py` 根据显式 meta 输出名称保存和读取新维数；
旧 meta 使用原来的 9 维名称。`policies/bc_policy.py` 新增 ChunkPolicy、PDPolicy 和
按 `output_representation` 选择的工厂；旧模型仍创建 BCPolicy。评估协议常量未改。
§1.2 的证据要求是最终版本全部 tests 通过、C3 seed 0 配方重训与原模型逐位相同，以及
k=1 训练矩阵 SHA256 与 C3 相同；重训产物只写入 `results/step6/smoke/`。

脚本第一个参数是阶段（默认 `all`），可用阶段为 `cache`、`pilot`、`promote`、`train`、
`models`、`offline`、`dev`、`hard`、`confirm`、`figures`。例如，单跑模型检查是
`bash results/step6/run_expG.sh models`。每阶段核对冻结记录与方案 SHA256；已完成的阶段
核对实现哈希后跳过。完整运行经校验后复用；没有 `summary.txt` 的中断 rollout 文件夹
删除重跑。门槛失败会写明停止位置与原因，并保持后续真实评估集未打开。CONFIRM 必须
通过 HARD 门槛，由链在一次性开启阶段传入 `--allow-confirm`；直接单跑该阶段同样受检查。

`WORKERS` 控制仿真进程（正式默认 22）；`JOBS` 控制训练并行请求（默认 1）。训练先运行
一个模型记录 peak RSS，随后按 `(可用内存 - 2 GiB) / peak RSS` 调度，最多 5 个进程；
任一进程超过 6 GiB 则停止。建议两条正式链顺序运行，`WORKERS=22 JOBS=1`，先保守保留
内存余量；Claude 可按实际峰值提高 JOBS，调度仍受上述上限约束。两条链同时运行时每条
用 `WORKERS=11`，并为两条链合计的训练内存留余量。

冻结与验收之后，由 Claude 运行正式长链：

```bash
WORKERS=22 JOBS=1 bash results/step6/run_expG.sh all >> results/step6/chainG.log 2>&1
WORKERS=22 JOBS=1 bash results/step6/run_expP.sh all >> results/step6/chainP.log 2>&1
```

执行者只运行预计 10 分钟以内的验证或短链。例如，以下接口使用 2 个训练 episode、
1 个验证 episode、1 个 epoch、2 个真实 pilot 闭环 episode，其余门槛阶段使用合成结果；
G、P 复用同一个独立 smoke 根目录，便于生成三者图表：

```bash
WORKERS=2 JOBS=1 bash results/step6/run_expG.sh all \
  --smoke-root results/step6/smoke/chain_example --smoke-epochs 1
WORKERS=2 JOBS=1 bash results/step6/run_expP.sh all \
  --smoke-root results/step6/smoke/chain_example --smoke-epochs 1
```

用另一个 smoke 根目录及 `--smoke-case dev_fail`、`dev_risk_fail`、`hard_fail`、
`confirm_fail`、`pilot_g_stop` 或 `pilot_p_stop` 检查相应自动停止分支。软件冒烟不打开
真实 DEV/HARD/CONFIRM；所有路径均位于指定 smoke 根目录。不要把 smoke 参数用于正式运行。

图表也可单独生成（Python 仍使用指定 pixi manifest）：

```bash
pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python \
  results/step6/gp_figures.py --root . --output results/step6 --experiment both
```

`--parts pilot,dev,tolerance,offline,smoothness` 选择图表，`--seeds 0,1 --n-boot 200`
用于小规模软件冒烟，`--require-complete` 在所需输入缺失时失败。第一次链结束时，未完成
另一实验的部分记录为 skipped；第二条链完成后，共同 DEV 表和容差图加入所有已完成的模型。
正式产物包括 `G_pilot.{txt,json}`、`fig_G_pilot.{png,pdf}`、`DEV_comparison_table.{txt,json}`、
`fig_tolerance_GP.{png,pdf}`、`tolerance_curve_GP.{txt,json}`、`P_offline_table.{txt,json}`、
`G_smoothness.{txt,json}`、`G/P_model_checks.json`、`G/P_MM_compare_<set>.{txt,json}`、
`G/P_mechanisms.{txt,json}`、`chainG/P_status.json` 和阶段事件日志。`data/test` 始终封存；
正式 G/P 缓存、模型和评估结果由 Claude 接续产生。
