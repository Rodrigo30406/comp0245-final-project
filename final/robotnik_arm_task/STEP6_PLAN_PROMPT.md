# 第 6 步方案：实验 G（action chunking + temporal ensembling）与实验 P（输出关节参考 + 固定 PD）

本文件是给执行者的自包含任务说明，也是实验 G、P 的预注册草稿（§6）。用户确认后才执行。
工作目录：`/home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task`
（git worktree，分支 worktree-eval-protocol；`data/`、`models/` 是指向主 checkout 的符号链接）。
Python 一律用：`pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python`。测试用 unittest（`tests/`，没有 pytest）。
机器：24 核、30 GB 内存（约 22 GB 可用）。并行仿真用 `--workers 22`（训练与仿真不要同时满载）。

## 0. 背景（只需知道这些；细节读代码）

- 任务、主模型、评估：同 `STEP5_DAGGER_PLAN_PROMPT.md` §0。主模型 C3 = `ego-novw-trig`（特征集 `ego_novw`，MLP 128x128，
  K = 16 份触发帧拷贝 + qd 抖动），`trained/ego-novw-trig_s<seed>.npz`。C3：DEV 60.5%、HARD 56.3%、CONFIRM 64.3%；
  主要失败 extended_not_reached 16-19%、circling 12-17%、held_10s 约 26%、侧翻 1-4%。实验 D、F 都没有被采纳。
- 动机：ACT（Zhao et al., RSS 2023, arXiv:2304.13705）的两点可以单独移植到现有 MLP 上，不需要 transformer 或 CVAE
  （专家是确定性脚本，没有多模态）：
  - G：action chunking（一次预测未来 k 个动作）+ temporal ensembling（每步查询，对同一时刻的多次预测加权平均）。
  - P：ACT 的动作空间是关节位置目标，由底层 PD 执行。对应到这里：模型输出专家的关节参考（q_des - q 与 qd_des），
    专家自己的 PD + pinocchio 重力补偿（`robotnik_arm_control.arm_controller` 最后几行）作为固定的、不学习的输出层换成力矩。
- 已有的诊断（report.txt `[Discussion] 与专家差距的结构性来源`）：专家的 q_des 是积分器，学习者看不到；离 target 1-2 cm 时
  可见的纠正信号约 0.1-0.3 N·m，C3 的力矩误差约 2.1 N·m（reach_move）。注意同一节第 2 条：PD 部分 kp(q_des-q) 与重力项反相关，
  PD 部分的标准差比力矩本身大，这是 P 的竞争假设 CP1 的依据。
- 模板：实验 D、F 的流水线（`results/step3/run_expD.sh`、`D_mechanisms.py`，`results/step5/run_expF.sh`、`F_mechanisms.py`、
  `f_common.py`）。照它们的结构写，不要重新发明。

## 1. 硬性规则（违反任何一条都要停下并在报告里说明）

1. 不打开 `data/test`（封存）。CONFIRM 只在 §6 的门槛通过后、用 `--allow-confirm` 打开一次（每个实验各一次）。
2. 不修改已有的结果、模型与冻结文件（`results/` 下已有的运行、`prereg_*_frozen.txt`、`trained/` 里已有的模型、
   `evaluate_policy.py` 的协议常量）。共享代码（`train_policy.py`、`bc_features.py`、`policies/bc_policy.py`、
   `analyze_rollouts.py`、`evaluate_policy.py`）只做向后兼容的添加（新参数默认关闭），并证明旧路径不变：现有 `tests/` 全部
   通过，且用 C3 配方重训 seed 0 与 `trained/ego-novw-trig_s0.npz` 逐位相同。
3. 不 commit、不 push、不改 git 配置。临时文件、日志、大的侧缓存不进版本库。
4. 冻结之后不改 §6 的任何规则。必须偏离时：记录偏离、原因与影响，按原规则也算一遍，两者都报告。
5. 只按 §6 的规则给出判定；不写超出规则的解读（解读由 Claude 审核时写）。
6. 被中断的运行：删除没有 summary.txt 的文件夹再重跑。
7. 分工（§7）：执行者（Codex）只做实现、验证、冻结与短的冒烟运行；长时间的训练与仿真链由 Claude 运行。

## 2. 共同设置（G、P 都与 C3 只差一个变量）

- 不变：split（`trained/data_split.json`）、stride 9 与每个 episode 的偏移、特征集 `ego_novw`、K = 16 触发帧拷贝与 qd 抖动
  （σ 的自动取法）、MLP_PARAMS（128x128、Adam lr 1e-3、alpha 1e-4、batch 200）、早停规则（patience 5、rel_tol 1e-3、最多 200 epoch）、
  float64、单线程。
- 新模型名：`ego-novw-trig-chunk`（G）、`ego-novw-trig-pd`（P），在 `train_policy.MODELS` 里登记。
- seed：DEV 用 0-9，HARD 用 0-4，CONFIRM 用 100-104（与 D、F 相同）。对照行与 D、F 完全相同（C3 s0-4 早停行 + stalled 重跑行，
  s5-9 不早停）；HARD、CONFIRM 的 C3 对照在门槛通过后按 F 的做法补 stalled 重跑。
- 所有新运行 `--no-early-stop`；DEV 加 `--save-traj --traj-every 10`。运行文件夹命名照 F（`results/<model>_s<k>_<set>/`）。
- 新文件放在 `results/step6/`；侧缓存放在 `results/step3/cache/chunk_d<Δ>/`、`results/step3/cache/pdref/`（带 manifest：
  代码 sha256、源文件 sha256、行数、规则说明）。

## 3. 实验 G：action chunking + temporal ensembling

### G1 标签（侧缓存）
- 每个训练行（episode e，录制步 s，即 stride-9 训练行与触发帧拷贝的源帧）的标签是 k = 10 个动作的块：
  y = [a(s), a(s+Δ), ..., a(s+9Δ)]，a = (v, w, tau1..7)（干净标签，与 `_episode_rows` 相同的取法），共 90 维。
  s + jΔ 超出 episode 末尾时用最后一步的标签（报告受影响的行比例，每个 Δ 分别报）。
- Δ ∈ {9, 27, 81} 步（奇数：低 z 时力矩标签逐步正负交替，偶数间隔会混叠，与 stride 的理由相同），即块的时长
  (k-1)Δ = 81、243、729 ms。块的第 0 段（j = 0）必须与 C3 的标签逐位相同。
- 验证集：早停只用 j = 0 段在验证集全部 1 kHz 步上的标准化 MSE（与 C3 的早停量完全相同，也避免 2.8M x 90 的内存）。
  另在 stride-9 的验证行上记录整块的验证损失（描述性）。

### G2 训练
- 输出 90 维，每一维按训练行的均值、标准差标准化，等权（底盘 / 机械臂的损失占比仍是 2:7，与 C3 相同）。
- 内存：每个进程预计 3-5 GB（sklearn 每个 epoch 打乱时复制数组）。第一个模型记录 peak RSS；同时训练的进程数按
  (可用内存 - 2 GB) / peak 取，最多 5。超过 6 GB / 进程就停下报告。
- `train_policy.py` 新增参数（默认关闭）：`--chunk-k`、`--chunk-delta`。k = 1 时训练矩阵的 sha256 必须与 C3 相同。

### G3 部署（`policies/bc_policy.py` 新增 ChunkPolicy，或在 BCPolicy 中按模型 meta 分支）
- 每个 1 kHz 步都查询模型，得到 10 段预测；在时刻 t 执行的动作是所有"指向 t"的预测的加权平均：
  来自查询时刻 t - jΔ 的第 j 段，j = 0..9（只用这些正好落在 t 上的预测，不插值）。
- 权重照 ACT：w_i = exp(-m i)，i = 0 是最旧的预测（j = 9）；m = 0.11，使最新与最旧的权重比为 e^{-0.99}，
  与 ACT 默认（k = 100、m = 0.01）相同。权重事先固定，不调。episode 开始时可用的预测不足 10 个，在可用的预测上归一化。
- (v, w) 与力矩一起平均，再换成轮速。reset 清空缓冲区。

### G4 验证（单元测试 `tests/test_chunk.py`）
- (a) 随机抽 1000 个训练行，块标签的每一段与录制数据中 s + jΔ 步的标签逐位相同（超出末尾的用最后一步）。
- (b) j = 0 段与 C3 训练标签逐位相同；k = 1 的训练矩阵 sha256 与 C3 相同。
- (c) temporal ensembling：常数模型时输出等于模型输出；用人造的已知序列核对权重、对齐（哪个查询的哪一段落在 t 上）与
  开头的归一化；k = 1 时 ChunkPolicy 与 BCPolicy 在 300 个录制步上一致（< 1e-9）。
- (d) 训练 / 部署一致：在一个录制 episode 上按顺序调用 act，与离线向量化的 ensembling 结果一致（< 1e-9）。

### G5 Δ 的校准（pilot，data/pilot，100 个 episode；不是 DEV / HARD / CONFIRM）
- 训练 Δ = 9、27、81 各 seed 0、1（6 个模型）；在 data/pilot 上跑完整模型（MM，`--no-early-stop`）。同时在 pilot 上跑
  C3 s0、s1 作为参照（如果 pilot 上还没有 C3 的 MM 运行）。
- 规则（冻结在 §6）：Δ* = pilot success（2 个 seed 平均）最高的 Δ；与最高值相差 ≤ 5 pp 的 Δ 中取最小的。
  停止规则：若 success(Δ*) < C3 pilot success - 10 pp，G 停止，不跑 DEV，报告"在 pilot 上三个块长都明显变差"。
- 选定 Δ* 后，seed 0、1 的校准模型直接作为正式模型（必须与用相同参数重训的结果逐位相同，抽查 seed 0）；再训练 seed 2-9 与
  100-104。
- 输出 `results/step6/G_pilot.txt/.json`、`fig_G_pilot.{png,pdf}`（pilot success 对块时长，C3 画在 0 ms）。

## 4. 实验 P：输出关节参考，固定的 PD + 重力补偿换成力矩

### P1 标签与固定输出层
- 输出 16 维：(v, w) + e = q_des_arm - q_arm（7）+ qd_des_arm（7）。q_des_arm、qd_des_arm 是录制的专家参考
  （`collect_data.py` 已录），q_arm = q_mes[:, 4:11]（与专家力矩公式用的是同一步的值）。
- 固定输出层 `pd_layer(e_hat, qd_des_hat, q_arm, qd_arm, q_pin, info)`，参数取 `ExpertPolicy` 用的同一个
  `ArmControllerParams`（不要复制常数）：
  tau = clip(kp e_hat + kd (qd_des_hat - qd_arm) + tau_g(q_pin)[6:13], ±max_torque)。
  tau_g 用 pinocchio `computeGeneralizedGravity`，q_pin 照 `BCPolicy.ee_position` 从 obs 构造。
  不对参考做任何截断（修订 1，见下）：专家的 ±max_q_lead 与关节限位截断只在 reach 分支里，已经包含在标签 e 里；
  固定层若再截断，就与专家在导航分支（以及关节 6 刚开始伸臂时）不同。
- 修订 1（2026-11-02，冻结之前；Claude 在执行者的 P2(b) 失败后写下，用户事先授权"直接跑、完成任务优先"，事后告知）：
  原方案要求固定层总是做 ±0.15 的 lead 截断和关节限位截断。P2(b) 在 train episode 4 上失败（最大差 18.4 N·m）：机械臂初始
  关节 6 为 1.0 rad，低于下限 1.134 + margin 0.05，专家在导航分支直接用 q_hold、不截断，固定层却把参考抬到 1.184 rad；
  在 reach 分支，限位截断也让 |e| 超过 0.15（10 个 episode 中 0.6% 的步）。Claude 的复核（只读 data/train 的 10 个 P2 episode）：
  去掉参考截断、只保留最后的 ±30 N·m 力矩截断，重建力矩与录制值的最大差为 1.9e-5 N·m（float32 量化）。改动只涉及固定层
  的定义，在任何 G/P 训练或闭环运行之前；P 的问题、变量与判定规则不变；竞争假设 CP3 相应改写。第一次失败的证据文件保留。
- 损失权重：标准化后机械臂的 14 维各乘 1/√2，使底盘 / 机械臂的损失占比与 C3 相同（2 : 7）。这只影响训练时的标准化，
  部署时反标准化抵消。

### P2 验证（单元测试 `tests/test_pd_layer.py`）
- (a) 精确：随机生成 200 个状态（关节角、速度、q_des、q_hold、目标；q_hold 与 q_des 不预先限制在 lead 或限位之内，
  包括关节 6 在限位以下的情形），直接调用 `arm_controller` 得到 (tau, q_des, qd_des)，pd_layer(q_des - q, qd_des, ...) 与 tau
  的最大差 ≤ 1e-9（reach_active 为真和假各 100 个）。
- (b) 录制数据（float32）：10 个 train episode（5 个干净、5 个带噪），pd_layer(录制的 e, qd_des) 与录制的 torque_arm 每步最大差
  ≤ 1e-3 N·m（与 F 的修订 1 同一容差与理由）。
- (c) 训练 / 部署一致（< 1e-9），同 G4 (d)。
- 这一步不通过，P 不做后面的任何事（G 可以继续）。

### P3 训练与离线描述
- 训练 seed 0；先在 data/pilot 上做冒烟测试（MM，seed 0，100 个 episode）：只检查有没有程序错误，不作判定。
  若 policy_error > 0、success < 10% 或侧翻 > 30%，停下查 bug 并报告（不算判定）。通过后训练 seed 1-9 与 100-104。
- 离线（描述性，验证集，每个 seed）：经 pd_layer 换算后的力矩 RMSE，按段（导航保持、reach_move、reach_hold）与
  z（< / >= 0.45 m），与 C3 对应 seed 的 `trained/*.offline.json` 对照；e 与 qd_des 各自的 nRMSE。

## 5. 评估与分析

- 脚本：`results/step6/run_expG.sh`、`run_expP.sh`（照 run_expF.sh：可续跑、已完成的跳过、阶段之间检查门槛），
  `results/step6/G_mechanisms.py`、`P_mechanisms.py`（照 F_mechanisms.py），输出 `G_mechanisms.txt/.json`、
  `P_mechanisms.txt/.json`、完整比较 `G_MM_compare_<set>.txt`、`P_MM_compare_<set>.txt`。
- 顺序：DEV（G、P 各 10 对 10）-> 各自的门槛 -> HARD（5 对 5）-> 门槛 -> CONFIRM（5 对 5，`--allow-confirm`）。
- 图表（给报告用）：
  - 表：DEV 上 C3、G、P 的 success 与各失败类（seed 平均、两层 bootstrap CI），以及 policy_ms（效率）。
  - 图：容差曲线（用 `results/step5/tolerance_curve.py` 的同一方法，DEV 上加 G、P）：看 5-20 mm 的差距有没有移动。
  - 图：G 的 pilot 校准曲线（G5）。
  - 表：P 的离线力矩 RMSE 对 C3（P3）。
  - G 的平滑度（描述性）：DEV 轨迹（每 10 步存一次）上 reach 阶段相邻存储点力矩差的绝对值的中位数，对 C3。

## 6. 预注册（冻结时把本节原文复制进 report.txt）

### 共同
- 统计单位 episode；两层 bootstrap `paired_diff` 与按 seed 的 Welch。G 与 P 同时对同一个对照做比较，所以条件 (1) 与镜像
  条件用 97.5% CI（Bonferroni，2 个比较）；其他 CI 为 95%。次要量在每个实验内做 Holm 校正，描述性。
- 判定（每个实验分别，主指标 success）：
  1. DEV：Δsuccess 的两个 97.5% CI 下界都 > 0。
  2. DEV：crash + left_or_overshot 的增加 ≤ +3 pp（点估计）。
  3. HARD：Δsuccess 点估计 > 0。
  4. CONFIRM：Δsuccess 点估计 > 0 且 bootstrap 95% 下界 > 0。
  - (1)-(4) 全部满足 -> 采纳。(1) 或 (2) 不满足 -> 该实验停止，HARD、CONFIRM 不为它打开，KEEP C3。
    (3) 或 (4) 不满足 -> KEEP C3，报告 DEV 与该集的差别。
  - 镜像：DEV 上两个 97.5% CI 上界都 < 0 -> 报告"该改动使 success 变差"。
  - G、P 都被采纳时：DEV 上 G 对 P 的配对差（seed 平均，两层 bootstrap 95%）不含 0 -> 主模型取较高者；含 0 -> 主模型取 G
    （G 不依赖"固定 PD 输出层是否符合作业要求"），P 作为另一个被采纳的变体报告。G + P 的组合不检验（时间不够，写进局限）。
- 次要量（Holm，描述性）：reach_2cm、reach_5cm、held_10s、arm_extended_toward、低处（z < 0.45 m）success、policy_ms。

### 实验 G（action chunking + temporal ensembling）
- 问题：一次预测未来 10 个动作、执行时对重叠的预测加权平均，能否提高 success？
- 唯一变量：输出从当前动作变成 10 段的动作块，部署加 temporal ensembling；块的间隔 Δ 按 G5 在 pilot 上选定。混杂：当前动作在
  损失中的份额降为 1/10；chunking 与 ensembling 不分开检验。
- 主假设 H-G：块预测与平均让底盘和机械臂的指令在时间上一致，减少闭环中的抖动和误差累积，所以底盘层面的失败减少。
- 机制预测（DEV，G 减 C3）：
  - MG1：circling 减少（点估计 < 0）。
  - MG2：held_10s 增加（点估计 > 0）。
  - MG3（来自"隐藏积分器"的诊断，预测不变）：extended_not_reached 的两个 95% CI 都在 ±7 pp 之内。
  - MG4：reach 阶段力矩的相邻差变小（点估计，描述性）。
- 竞争假设：
  - CG1 延迟：最旧的预测权重最大，反应变慢，过冲与 circling 增加；pilot 上大的 Δ 变差。
  - CG2 无效：相邻观测的预测误差高度相关，平均几乎不减小误差；success 与失败类都不变。
  - CG3 当前动作变差：j = 0 段的验证损失高于 C3（同一个量，可直接比），精度变差，extended_not_reached 增加。
  - CG4 积分器诊断不对：extended_not_reached 明显减少（MG3 不成立）。这会否定 Discussion 中"最后 1-2 cm 来自隐藏状态"的说法。
- Δ 的选择规则与停止规则：G5（冻结时连同本节一起冻结）。

### 实验 P（输出关节参考 + 固定 PD）
- 问题：让模型输出专家的关节参考（q_des - q、qd_des），由专家的 PD + 重力补偿换成力矩，能否提高 success？
- 唯一变量：机械臂的输出表示（力矩 -> 参考 + 固定输出层）；底盘输出、特征、数据、K 都不变。混杂：输出维数 9 -> 16
  （用 1/√2 的权重保持底盘 / 机械臂的损失占比）。
- 主假设 H-P：固定层给出精确的刚度 kp、阻尼 kd 与重力补偿，参考误差 δe 只造成约 δe 的位置误差；C3 在 target 附近的有效刚度
  是学出来的、未知的。所以最后几厘米更准，侧翻更少。
- 机制预测（DEV，P 减 C3）：
  - MP1：extended_not_reached 减少（两个 95% CI 上界都 < 0）。
  - MP2：crash 减少（点估计 < 0）。
  - MP3：circling 不变（两个 95% CI 都在 ±7 pp 之内；底盘输出没变）。
  - MP4：离线 reach_hold 的力矩 RMSE 低于 C3（10 个 seed 的中位数，描述性）。
- 竞争假设：
  - CP1 放大：e 的误差经 kp = 100 放大（0.01 rad = 1 N·m），而 PD 部分本身的方差比力矩大（Discussion 第 2 条），离线力矩 RMSE
    不低于 C3，success 不升。
  - CP2 积分器仍然看不到：e 的"领先"部分依赖历史，机械臂仍停在离 target 1-2 cm 处，extended_not_reached 不变。
  - CP3 触发处的跳变：专家开始伸臂时关节 6 的参考被限位截断从 1.0 抬到约 1.18 rad（e 跳变约 0.18 rad，约 18 N·m），
    这个跳变取决于锁存的 reach 状态；学习者若把它学平，伸臂的起始会变化。看 arm_extended_toward 与 parked_no_arm。

## 7. 分工与时间（用户 2026-11-02 确认：不设硬截止，完成任务优先）

- 执行者（Codex）：§8 的第 1、2 步全部，以及第 3 步需要的所有脚本（缓存构建、训练、pilot、选 Δ、门槛、HARD、CONFIRM、
  分析与图表都写进 `run_expG.sh` / `run_expP.sh` 的阶段里）。只运行短任务：单元测试、向后兼容检查、在少量 episode 上
  的冒烟（例如 2 个 episode 的缓存、1-2 个 epoch 的训练、2 个 episode 的 pilot 运行），证明链的每个阶段都能跑通，然后把冒烟
  产生的临时文件删掉或放在 `results/step6/smoke/`（不能与正式运行的路径冲突）。任何预计超过 10 分钟的命令都不要运行，
  写进 REPORT_GP.md 的"续跑命令"里交给 Claude。
- Claude：审核执行者的改动后，运行 `bash results/step6/run_expG.sh all >> results/step6/chainG.log 2>&1` 与
  `run_expP.sh all`（顺序或并行由 Claude 按内存决定），然后审核结果、写解读。
- 链脚本要求：可续跑（已完成的阶段与运行跳过，中断的运行文件夹自动删除重跑）；每个阶段开始时核对冻结文件的 sha256；
  按 §6 的门槛自动停止（停在哪里、为什么，写进日志与 `results/step6/chain*_status.json`）。

## 8. 执行顺序与冻结

1. 读代码（run_expF.sh、F_mechanisms.py、f_common.py、train_policy.py 的缓存与 train-many、bc_policy.py、evaluate_policy.py、
   robotnik_arm_control.py、analyze_rollouts.py），写 G1、G3、P1 与测试 G4、P2，全部通过；向后兼容检查（§1.2）。
2. 冻结：把 §6 原文复制进 report.txt，新节标题 `[Methodology] 第 6 步预注册：实验 G（action chunking）与 P（关节参考输出）
   （2026-11-0x 写下，在 G、P 的任何 pilot / DEV / HARD / CONFIRM 运行之前；冻结记录 results/step6/prereg_GP_frozen.txt）`，
   放在第 5 步结果节之后；把该节文本与本文件的 sha256、冻结时间写进 `results/step6/prereg_GP_frozen.txt`。之后每个阶段
   开始时核对。
3. 链（由 Claude 运行，见 §7）：G5 校准与 P 冒烟 -> 训练 -> 模型检查（meta 的特征集、K、σ、Δ、输出表示、代码与缓存
   sha256）-> DEV -> 门槛 -> HARD -> 门槛 -> CONFIRM -> 分析与图表。日志 `results/step6/chainG.log`、`chainP.log`。
   仿真阶段默认 `--workers 22`；两条链并行时各用 11（脚本接受环境变量 WORKERS）。训练并行数接受环境变量 JOBS。
4. README.md：补"第 6 步"的说明（新文件、改动、怎么运行、输出），注明新写与修改的代码；P 的固定输出层复用了专家的 PD 律
   与重力补偿，要写明。

## 9. 交付与汇报

写 `results/step6/REPORT_GP.md`（给审核者看，用中文）：
- 新建与修改的文件；对共享文件的每处改动与"旧路径不变"的证据。
- G4、P2 的验证结果；G5 的 pilot 表与选定的 Δ*；P 的冒烟结果；内存与训练时间；模型检查。
- 每个阶段的起止时间、运行数、harness_error / policy_error、中断与重跑。
- §6 每条判定的数字（摘自 *_mechanisms.txt）与最终判定；所有偏离及其影响。
- 没有完成的部分与续跑命令。
可在 report.txt 写结果节，标题带"草稿（待 Claude 审核）"，只写数字与按规则的判定，不写解读。
ACT 的参考文献由 Claude 核对出处后再加进 report.txt 的 [References]。
