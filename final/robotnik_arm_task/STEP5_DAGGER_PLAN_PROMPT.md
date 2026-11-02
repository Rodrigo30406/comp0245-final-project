# 第 5 步方案：F0 容差曲线 + 实验 F（DAgger 一轮，影子专家标注）

本文件是给执行者（Codex）的自包含任务说明，也是实验 F 的预注册草稿（§3）。用户确认后才执行。
工作目录：`/home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task`
（git worktree，分支 worktree-eval-protocol；`data/`、`models/` 是指向主 checkout 的符号链接）。
Python 一律用：`pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python`。测试用 unittest（`tests/`，没有 pytest）。
机器：24 核、30 GB 内存、磁盘约 300 GB 空闲。并行仿真用 `--workers 22`（训练与仿真不要同时满载）。

## 0. 背景（只需知道这些；细节读代码）

- 任务：行为克隆 Robotnik 底盘 + Kinova 7 关节机械臂的伸臂到达。策略输入是当前观测，输出底盘 (v, w)（再换成 4 个轮速）
  和 7 个关节力矩。主模型 C3 = `ego-novw-trig`（特征集 `ego_novw`，MLP 128x128，每个训练 episode 加 K = 16 份触发帧拷贝），
  模型文件 `trained/ego-novw-trig_s<seed>.npz`，部署 `policies/bc_policy.py:make_policy`。
- 评估：`evaluate_policy.py`（协议 v1.1，success = 末端在 T_MAX 内进入 target 10 mm），比较与统计：`analyze_rollouts.py`
  （两层 bootstrap `paired_diff`、按 seed 的 Welch `seed_t_test`、失败分类 `classify_run`）。实验 D 的完整流水线
  `results/step3/run_expD.sh` + `results/step3/D_mechanisms.py` 是本步最直接的模板：照它的结构写，不要重新发明。
- C3 的表现：DEV 60.5%、HARD 56.3%、CONFIRM 64.3%，专家 97-99%。主要失败：extended_not_reached 16-19%（末端停在离 target
  1-4 cm 处）、circling 12-17%、到达后底盘蠕动。report.txt 的 `[Discussion] 与专家差距的结构性来源` 说明了原因：专家的
  参考 q_des 是一个积分器（学习者看不到），离 target 1-2 cm 时无记忆可见的纠正信号（约 0.1-0.3 N·m）远小于模型的力矩误差
  （reach_move RMSE 约 2.1 N·m）。
- 专家（`evaluate_policy.ExpertPolicy`）是有内部状态的：底盘导航 FSM（`BaseNavState`）和机械臂参考 `q_des_arm`
  （每步积分 qd_des，限制在 q ± 0.15 rad）。它的 `act(obs)` 只依赖观测序列与 `reset(target, info)` 的 info，所以沿任意一条
  1 kHz 轨迹按顺序重放 `act`，就得到"影子专家"在学习者状态上的标签（专家内部状态沿学习者的轨迹演化）。这就是本步 DAgger
  的标注方式，不需要改评估 harness。
- 记录格式：`evaluate_policy.py --save-traj --traj-every 1` 保存每一步的 `TRAJ_FIELDS`
  （t, base_pos, base_ori, base_bearing, q_mes, qd_mes, ee_pos, wheel_cmds, torque_arm）。训练标签取法见
  `train_policy._episode_rows`：(v, w) = `bc_features.wheels_to_vw(wheel_cmds)`，力矩 = `torque_arm`（专家的干净指令）。

## 1. 硬性规则（违反任何一条都要停下并在报告里说明）

1. 不打开 `data/test`（封存）。CONFIRM 只在 §3 的门槛通过后、用 `--allow-confirm` 打开一次。
2. 不修改已有的结果、模型与冻结文件：`results/` 下已有的运行文件夹、`results/step3/prereg_*_frozen.txt`、`trained/` 里已有的
   模型、`evaluate_policy.py` 中的协议常量。对共享代码（`train_policy.py`、`bc_features.py`、`analyze_rollouts.py`、
   `evaluate_policy.py`）只做向后兼容的添加（新参数默认关闭），并证明旧路径不变：现有 `tests/` 全部通过，且用 seed 0 的
   C3 配方重训一次的模型数组与 `trained/ego-novw-trig_s0.npz` 逐位相同（若重训不可逐位复现，改为证明新代码在默认参数下
   产生的训练矩阵 sha256 与旧代码相同，并说明）。
3. 不 commit、不 push、不改 git 配置。临时文件与日志不进版本库（本来就未提交，保持现状）。
4. 冻结之后不改 §3 的任何规则。执行中发现必须偏离时：记录偏离、原因与影响，按原规则也算一遍，两者都报告。
5. 只按 §3 的规则给出判定；不写超出规则的解读（解读由 Claude 审核时写）。
6. 每一个被中断的运行：删除没有 summary.txt 的文件夹再重跑（与之前的做法相同）。

## 2. 任务

### F0（描述性，只用已有运行，不需要新仿真）：成功率对容差的曲线

- 新脚本 `results/step5/tolerance_curve.py`：对 DEV、HARD、CONFIRM，读专家、`ego-novw`（MM）、`ego-novw-trig`（C3）的
  已有运行，用 rollouts 的 `min_error` 列计算 P(min_error < τ)。τ 取 2-100 mm 的对数网格画连续曲线，并在表中列出
  τ = 5、7.5、10、15、20、30、50、100 mm。每个模型给 seed 平均与两层 bootstrap 95% CI（`analyze_rollouts.boot2` 的同一方法）。
  对照行的构造与 `D_mechanisms.py` 相同（C3 s0-4 的早停运行中 stalled 的 episode 用 `results/step3/nostall/` 的重跑行替换）；
  先核对 `min_error` 的定义（是否只到 v1 停止点）并在输出中写明。
- 修订 2（2026-11-02，只改 F0 这一描述性部分，不涉及冻结的 §3）：上一条把"替换 stalled 行"与"等于已发表的 60.5%"同时
  作为要求，两者矛盾（替换后 DEV 为 60.8%，正是 report.txt C3 结果第 2) 条的敏感性值）；HARD、CONFIRM 也没有现成的重跑行。
  改为：F0 的主曲线用每个运行的原始行（与已发表的 success 一致，检查 DEV 60.5%、HARD 56.3%、CONFIRM 64.3%，MM 0.3 / 0.4 /
  0.3%）；DEV 另加一条"stalled 行替换为 nostall 重跑行"的敏感性曲线，检查 τ = 10 mm 为 60.8%。HARD、CONFIRM 不做替换。
  F5 中 C3 HARD / CONFIRM 对照的 stalled 重跑照原计划在各自门槛通过之后进行，与 F0 无关。
  一般规则：描述性的一致性检查如果出现差别，而差别能由 report.txt 中已记录的数值完全解释，就记录原因后继续，不必停止；
  验证测试、§1 的规则与 §3 的判定照旧，出问题必须停。
- 输出：`results/step5/tolerance_curve.txt/.json`、`results/step5/fig_tolerance.{png,pdf}`。检查：τ = 10 mm 的值必须与已有的
  success 一致（DEV 60.5%、HARD 56.3%、CONFIRM 64.3%；MM 0.3/0.4/0.3%），τ = 20、50 mm 与 reach_2cm、reach_5cm 一致；
  不一致就先查清原因。

### F1：影子专家重放（`dagger_relabel.py`，新文件）

- `replay_expert(traj, episode)`：用 `evaluate_policy.setup_simulation(init_xy, init_yaw, target=target)` 取 info（取完断开
  pybullet），建 `ExpertPolicy`，`reset(target, info)`，按顺序对每一步调用 `act(obs_k)`（obs 由 traj 字段与 target 组成，
  与 harness 传给策略的一样），返回每步的影子 wheel_cmds (4)、torque_arm (7)，以及诊断量 nav_done、q_des_arm。
- 必须先通过的验证（单元测试 `tests/test_dagger_relabel.py`）：在 `data/train` 的 10 个 episode 上（5 个干净、5 个带 DART
  噪声的；只用 split 的 train_episodes）重放录制的观测序列，影子标签与录制的 wheel_cmds、torque_arm 每一步相同
  （最大绝对误差 ≤ 1e-9；如有更大差别，查清原因，例如 init_q_mes 与录制的第 0 步是否一致，噪声是加在执行指令上还是标签上）。
  这一步不通过，不做后面的任何事。
- 修订 1（2026-11-02，冻结之前，用户确认）：上面 ≤ 1e-9 的要求假定录制是 float64，但 `collect_data.py` 把观测与标签存成
  float32，第一次验证（`results/step5/F1_verification.json`）因此不可能通过：力矩最大误差 7.9e-5 N·m、轮速 5.2e-6 rad/s、
  nav_done 每步相同，与 float32 量化相符。验证改为两部分，两部分都通过才继续：
  (a) 精确验证（与 F2 同一条路径）：`evaluate_policy.py --policy expert --eval-set data/pilot --episodes <前 5 个>
      --save-traj --traj-every 1 --workers 5`，输出到 `results/step5/F1_expert_pilot/`；先确认保存的轨迹是 float64，
      再用 `replay_expert` 重放，影子 wheel_cmds、torque_arm 与轨迹中的每一步最大绝对误差 ≤ 1e-9，nav_done 与专家
      在 harness 中的状态一致（若轨迹没有 nav_done，就核对影子 nav_done 的第一步与 rollouts.csv 的 arm_move 无矛盾，并说明）。
  (b) 录制数据（float32）：仍用原来的 10 个 train episode，容差为力矩 ≤ 1e-3 N·m、轮速 ≤ 1e-4 rad/s、nav_done 每步相同。
      依据：1e-3 N·m 比模型的力矩 RMSE（约 2 N·m）小三个数量级，比离 target 1-2 cm 时的纠正信号（约 0.1 N·m）小两个数量级。
  第一次验证的文件保留不动，新结果用新的文件名。

### F2：学习者 rollout（训练分布的初始条件）

- episode：从 `trained/data_split.json` 的 `train_episodes` 中，用 `numpy.random.default_rng(5)` 无放回抽 400 个
  （绝不用 val episode），排序后轮流分给 C3 seed 0-4（每个 seed 80 个）。
- 运行：`evaluate_policy.py --policy policies/bc_policy.py:make_policy --policy-arg trained/ego-novw-trig_s<k>.npz
  --eval-set data/train --episodes <列表> --no-early-stop --save-traj --traj-every 1 --workers 22`，输出到
  `results/step5/rollouts/c3_s<k>_train/`（放在 step5 下，避免被 `analyze_rollouts.discover_runs` 当成评估运行）。
  先确认 `data/train` 能作为 `--eval-set` 读入，episode 编号与 split 的编号是同一套。
- 先做 pilot：2 个 episode 跑完 F2 -> F3，检查文件大小、耗时、标签范围，再跑全部 400 个。

### F3：DAgger 训练行（侧缓存）

- 对每个 rollout 的整段运行（含 reach 之后的 10 s），取步 offset::9（每个 rollout 一个 offset，
  `default_rng(6).integers(0, 9)`，按 rollout 的排序顺序），特征 = `bc_features.build_features("ego_novw",
  measured_signals(...))`，标签 = 影子的 (v, w) 与 torque_arm。写到 `results/step3/cache/dagger1/`，带 manifest
  （每条轨迹的 sha256、代码 sha256、行数、规则说明），格式与已有侧缓存一致。
- 描述性检查（写进 `results/step5/F_data.txt`，在任何 F 模型的 DEV 结果之前）：行数；影子 nav_done 的行比例；
  影子力矩标签在 ±30 N·m 截断处的比例、|tau| 的 p50/p99，与专家训练行对比；C3 在这 400 个 rollout 上的 success 与失败类
  （只作背景）。
- 轨迹文件很大（每个可达约 20 MB）：缓存与 manifest 写完、验证通过后可以删除 `results/step5/rollouts/*/traj`，但保留
  rollouts.csv、summary.txt、日志。

### F4：训练 `ego-novw-trig-dag1`

- `train_policy.py` 加向后兼容的参数 `--extra-rows <侧缓存目录>`：在原训练行与触发帧拷贝之后追加这些行，不加权，用原
  stride-9 训练行的均值与尺度标准化（与触发帧拷贝相同）；验证集不变（仍是专家的 val episode）；meta 记录侧缓存 manifest
  的 sha256 与行数。其他一切与 C3 相同（`--trigger-copies 16`、自动 σ 抖动、MLP_PARAMS、早停规则）。
- 训练 seed 0-9 与 100-104：`train_policy.py train-many --model ego-novw-trig-dag1 --seeds 0-9,100-104
  --trigger-copies 16 --extra-rows results/step3/cache/dagger1 ...`（其余参数照 `run_expD.sh` 中 C3 对照的写法）。
- 模型检查（照 `run_expD.sh` 的检查）：15 个模型 meta 的特征集、K、σ、代码与缓存 sha256 都正确。

### F5：评估与分析（`results/step5/run_expF.sh` + `results/step5/F_mechanisms.py`，照 run_expD.sh / D_mechanisms.py）

- 所有新运行 `--no-early-stop`；DEV 加 `--save-traj --traj-every 10`。
- DEV：dag1 s0-9 对 C3 s0-9（C3 的对照行与实验 D 的完全相同：s0-4 早停行 + stalled 重跑行，s5-9 已是不早停）。
- 门槛通过后 HARD：dag1 s0-4 对 C3 s0-4（已有的 C3 HARD 运行；先查它们是否早停，若是，stalled 的 episode 按 D 的做法用
  `--no-early-stop` 重跑后替换）。
- 门槛通过后 CONFIRM（`--allow-confirm`，只一次）：dag1 s100-104 对 C3 s100-104（同上处理）。
- `F_mechanisms.py` 输出 `results/step5/F_mechanisms.txt/.json` 与完整比较 `F_MM_compare_<set>.txt`，内容按 §3 逐条给数字
  与判定。

## 3. 实验 F 预注册（冻结时把本节原文复制进 report.txt）

- 问题：只用专家自己访问过的状态训练（C3）时，学习者停在离 target 1-4 cm 处、绕圈、到达后蠕动。用影子专家在学习者
  访问的状态上打标签（DAgger 一轮，β = 0，即 rollout 完全由 C3 执行），能否提高 success？
- 唯一变量：训练集多出 F3 的 DAgger 行（其余配方与 C3 相同）。混杂（不分开）：行数增加也改变每个 epoch 的 Adam 步数与
  早停时机；没有"多加同样数量的专家行"的对照。
- 主假设 H-F：影子专家的积分器在学习者停住时继续累积，标签里带有当前缺少的近处纠正力矩与偏离后的恢复动作，所以
  success 提高。
- 机制预测（DEV，dag1 减 C3，seed 平均，两层 bootstrap 与 Welch）：
  - M1 精度：extended_not_reached 减少（两个 CI 上界都 < 0）。
  - M2 分布偏移：circling 减少（点估计 < 0，描述性）。
  - M3 保持：held_10s 增加（点估计 > 0，描述性）。
  - M4 风险：crash（侧翻）与 left_or_overshot 之和的增加 ≤ +3 pp（点估计）。
- 竞争假设：
  - C1 过冲：影子标签在停住处很大（积分器领先最多 0.15 rad，约 15 N·m），学会后过冲或振荡，表现为 left_or_overshot、
    crash 增加，success 不升。
  - C2 标签冲突：同一观测对应不同的标签（取决于影子的历史），条件均值变平，没有改善；专家验证集损失上升。
  - C3 只是数据更多：改善来自行数与训练步数，与学习者状态无关（本实验分不开，写进局限）。
- 判定（主指标 success，统计单位 episode，两层 bootstrap `paired_diff` 与按 seed 的 Welch，95%）：
  1. DEV：Δsuccess 的两个 CI 下界都 > 0。
  2. DEV：M4 成立（crash + left_or_overshot 的增加 ≤ +3 pp）。
  3. HARD：Δsuccess 点估计 > 0。
  4. CONFIRM：Δsuccess 点估计 > 0 且 bootstrap 下界 > 0。
  - (1)-(4) 全部满足 -> 采纳 dag1 为主模型。(1) 或 (2) 不满足 -> 停止，HARD 与 CONFIRM 不为 F 打开，结论 KEEP C3。
    (3) 或 (4) 不满足 -> KEEP C3，并报告 DEV 与该集的差别。
  - 镜像：DEV 上两个 CI 上界都 < 0 -> 报告"DAgger 一轮使 success 变差"。
- 次要量（Holm 校正，描述性）：reach_2cm、reach_5cm、held_10s、arm_extended_toward；target z < 0.45 m 分层的 success；
  专家验证集损失（C2 的检查）。
- seed 数：DEV 10 对 10（与 D 相同，C3 的 seed 间标准差约 7 pp）；HARD、CONFIRM 各 5 对 5。

## 4. 执行顺序与冻结

1. 读代码（run_expD.sh、D_mechanisms.py、train_policy.py 的缓存与 train-many、evaluate_policy.py、analyze_rollouts.py），
   写 F1 并通过验证测试。
2. 冻结：把 §3 原文复制进 report.txt，新节标题 `[Methodology] 第 5 步预注册：实验 F（DAgger 一轮）（2026-11-02 写下，在
   F 的任何 DEV/HARD/CONFIRM 运行之前；冻结记录 results/step5/prereg_F_frozen.txt）`，放在
   `[Discussion] 与专家差距的结构性来源` 之后；把该节文本与本文件的 sha256、冻结时间写进
   `results/step5/prereg_F_frozen.txt`。之后每个阶段开始时核对这些 sha256。
3. F0（可与 F2 并行，F0 不占仿真）。
4. F2 pilot -> F2 全部 -> F3 -> F4 -> 模型检查 -> F5 DEV -> 门槛 -> HARD -> 门槛 -> CONFIRM -> F_mechanisms 最终输出。
   长任务用 `nohup` 跑成可续跑的链（照 run_expD.sh：已完成的运行跳过），日志写到 `results/step5/chainF.log`，然后轮询
   日志直到完成。如果你的会话不能等这么久，把链启动好、写清续跑命令再结束。
5. README.md：补"第 5 步"的说明（新文件、改动、怎么运行、输出），注明哪些代码是新写的、哪些是修改的。

## 5. 交付与汇报

写 `results/step5/CODEX_REPORT_F.md`（给审核者看，用中文）：
- 新建与修改的文件列表；对共享文件的每处改动与"旧路径不变"的证据（测试输出、sha256 或逐位比较）。
- F1 验证结果（每个 episode 的最大误差）；F0 的一致性检查；F3 的描述性检查；模型检查。
- 每个阶段的开始与结束时间、运行数、harness_error / policy_error 数、被中断与重跑的运行。
- §3 每条判定的数字（直接摘自 F_mechanisms.txt）与最终判定；所有偏离及其影响。
- 没有完成的部分与续跑命令。
可在 report.txt 写一个结果节，标题必须带"草稿（Codex，待 Claude 审核）"，只写数字与按规则的判定，不写解读。
