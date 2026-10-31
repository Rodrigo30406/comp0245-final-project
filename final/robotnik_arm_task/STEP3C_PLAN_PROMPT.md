# 第 3 步续：实验 C3（触发帧加权）方案 prompt

2026-10-30 定稿，过程如下：
- 用户先选了方案 A（v̂、ŵ 只给机械臂）。
- 3 个独立审查（统计设计、对照代码的可行性、价值与风险）用离线证据否定了 A 的前提（见第 2 节 C0）。
- 用户随后确认改做 C3。

本文件交给新会话执行。按用户的工作方式：先读材料、实现代码与测试；把预注册写进 report.txt 并冻结，然后才运行。
遇到方案没有覆盖的情况，或者第 4 节写着"停下"，就停下来向用户提方案，等确认后再继续。

## 0. 背景与规则

- 工作目录：/home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task
  - 本地分支：worktree-eval-protocol。
  - 小组仓库是 remote team。第 3 步在 team 的 feature/step3-bc（e669b87、6dcf93a），叠在 feature/eval-protocol 之上，两者都还没合进 main。
- Python：pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python ...
  - 有 numpy、scipy、sklearn 1.9.1、matplotlib、pinocchio、pybullet；没有 torch、pandas、pytest。
  - 单元测试用 unittest：python -m unittest discover -s tests（目前 147 个测试，全部通过）。
  - 机器 24 线程、30 GiB。evaluate_policy --workers 20 时：DEV 约 10-12 分钟，加 --no-early-stop 且多数失败时约 12.5 分钟；HARD 约 7 分钟；CONFIRM 约 12 分钟。
- 开始前先读：
  - report.txt 中第 3 步的各节：协议 v1.1、训练管线、失败分类、统计方法、预注册、实验 A 结果、实验 B 结果（中文）。
  - results/step3/summary.txt、B_ego-novw_dev.txt、A_ego_vs_ego-novw_dev.txt。
  - results/step3/C0/README.md 及其 *.txt。
  - 代码：train_policy.py（train_mlp、cache、split、offline）、bc_features.py、policies/bc_policy.py、policies/hybrid.py、analyze_rollouts.py、results/step3/mechanism_checks.py、results/step3/run_expB.sh。
- 队友 Rodrigo 的分支 team/rodrigo：
  - 读法：git fetch team 之后用 git show team/rodrigo:<path> 读，不要 checkout。在子目录里列文件要用 git ls-tree -r --full-tree。
  - 2026-10-30 的更新已经读过，见第 2 节。开始前再 fetch 一次，只看比 95de559 新的提交。
- 评估集：DEV（seed 200）做所有决定；HARD（seed 201）；CONFIRM（seed 202）只在采纳前复现一次（需要 --allow-confirm）；test（seed 100）封存，不碰。
- 工作方式（与第 3 步相同）：
  - 先写预注册：假设、竞争假设、机制预测、主指标与地板规则、采纳规则。
  - 一次只改一个变量；在同一批 episode 上配对；多 seed；给 CI。
  - 预注册之前看过的东西如实记录。
  - 值得写进 report 的内容追加到 report.txt，并告诉用户追加了哪些。用中文回答用户。
- 提交规则（用户 2026-10-30）：
  - 只在用户要求时 commit / push。
  - 临时目录（results/tmp_*）和运行日志（*.log、*.err、*.stdout、logs/）不提交，.gitignore 已覆盖。
  - 训练好的 trained/*.npz 可以提交。
  - 新的缓存放在 results/step3/cache/ 下（.gitignore 只认名为 cache 的目录）。
  - 推送方式：在临时 worktree 里，基于 team 的上一个分支 cherry-pick，推成新的 feature/<task>，用完删掉临时 worktree。
  - 提交前用 git status 检查有没有大文件或日志。
- demo（课内进度演示，2026-10-30 至 11-06）：在 C3 有结果之前，用现有结果演示：
  - (i) ego-novw 全模型：底盘开到环上，机械臂不伸，绕圈；
  - (ii) ME 格：学习的底盘 + 专家机械臂，96.7%，标明是诊断格；
  - (iii) C0 的机制解释："接着伸"对"开始伸"。
  GUI 命令：evaluate_policy.py ... --episodes <i> --gui --out-dir results/tmp_demo。

## 1. 已有的模型与运行（对照组全部已有）

- 模型（同一划分 trained/data_split.json，同一训练行，同一超参数）：
  - trained/ego-novw_s{0..4}.npz（主线，实验 A 采纳）、ego-novw_s{100..104}.npz；
  - ego_s{0..4,100..104}.npz。
- 运行（协议 v1.1）：

  | 格子 | 目录 | 运行方式 | 结果 |
  |---|---|---|---|
  | MM（全模型） | results/ego-novw_s{0..4}_{dev,hard}、ego-novw_s{100..104}_confirm | 早停 | success < 1% |
  | MM 的 stalled 重跑 | results/step3/nostall/ego-novw_s{1..4}_dev | --no-early-stop | success、reach_5cm、arm_extended、parked 全部不变 |
  | EM（专家底盘 + ego-novw 机械臂） | results/B-EM_ego-novw_s{0..4}_dev | --no-early-stop | 0.1%，1500/1500 不伸臂 |
  | ME（模型底盘 + 专家机械臂） | results/B-ME_ego-novw_s{0..4}_dev | | 96.7% |
  | EE | results/B-EE_dev、results/expert_v11_{dev,hard,confirm} | | 专家参照 |

## 2. 依据（预注册之前已看到，写进 report.txt）

脚本和输出都在 results/step3/C0/（README.md 说明每个脚本），从任务目录运行。

1) 实验 B（report.txt）
   - EM 0.1%：模型机械臂在 1500/1500 个 seed x episode 中从来没有伸臂，末端位移最大 5.3 cm（与导航时保持姿态的晃动相同）。
   - ME 96.7%：其中 85.6% 由阈值触发，即模型底盘自己到达了专家的停靠位姿。
   - MM 的失败中，99.9% 可归因于机械臂。
2) 方案 A 被否定（offline_trigger_stats.txt、probe_trigger.txt、probe_time.txt、probe_jitter.txt）
   - 此前写过"底盘静止的 8,802 帧 100% 是 reach"。复查发现其中 8,801 帧机械臂已经在动（max|qd_arm| >= 0.2 rad/s）。
   - 触发那一帧（nav_done 的第一步 k0）底盘还在转：|ŵ| 中位 0.138 rad/s。第一次静止在 nav_done 之后中位 33 ms。
   - 284 个验证 episode 中，没有一个是"底盘先静止、机械臂后开始动"。
   - 机械臂安静、看起来已停好的帧，只有 0.2-2.1% 是 reach。
   - 结论：v̂、ŵ 不是机械臂的触发线索。
3) 两类网络只会"接着伸"（probe_time.txt）
   - 响应分数 r：以 k0-1 为基线，预测力矩变化在标签跳变方向上的比例；0 = 保持，1 = 标签。
   - 在录制的 k0 帧，ego 和 ego-novw 的 r 都是 0.00；从 k0+1 起 r = 0.76-0.96，之后约 1.0。
   - EM 稳态（合成：专家停靠后的底盘位姿、v̂ = ŵ = 0、机械臂在 k0-1 的保持姿态、qd = 0）：ego 的 r 中位 0.003-0.007，ego-novw 0.001-0.002。
   - 推动对照：同一稳态下，把机械臂速度换成 k0+1 ms 的录制值，r = 0.67-0.87。机械臂一旦开始动，就会自我维持（机械臂一侧的照抄）。
4) 触发帧在数据里只占一帧（k0labels.txt、footprint.txt）
   - k0 的标签：v = w = 0，力矩跳变中位 27.9 N·m（p10 21.3）。k0-1 的标签：|w| 0.150。
   - k0 时：环距离中位 0.074 m（最大 0.180），|beta| 中位 0.032 rad。
   - 训练行（步长 9）里，"nav_done 且机械臂在保持姿态、max|qd| < 0.1"的只有 147 行（0.008%），其中底盘静止的 1 行。
   - 与 k0 相像但标为保持的帧（在环附近、朝向、机械臂在保持姿态）每个 episode 约 74 行（步长 9）。
   - 这些相像帧上，MSE 回归的条件均值就是"保持"。
5) 绕圈对总成功率的影响（circling.txt）：把 MM 的每个 seed x episode 和 ME 格的同一个单元对照：
   - MM 的 483 个 circling 中：337 个（69.8%）在 ME 格里阈值触发后成功；108 个靠 25 s 超时成功；38 个失败。
   - MM 的 815 个 parked_no_arm 中：734 个（90.1%）阈值触发后成功。
   - 即：绕圈大多发生在底盘已经到过停靠位姿之后，是"机械臂不伸 -> 底盘没有'已到达'信号"的后果。
   - 触发修好后，仍有约 10% 的 episode 是底盘自己的问题。
6) 反面证据（check455.txt）：底盘自己停靠并朝向 target（< 6 度）时，伸臂的次数：

   | 模型 | DEV | HARD |
   |---|---|---|
   | ego | 55 个中 4 个 | 70 个中 0 个 |
   | ego-novw | 177 个中 4 个 | 115 个中 3 个 |

   对于 ego 的 55 个：48 个来自 seed 3、4；22 个的朝向误差在专家 orient 容差之外。
7) Rodrigo 2026-10-30 的更新（docs/BLACKBOX_PLAN_20261027.md §6f-§6i）：
   - 随机起点的数据能教会底盘停住，但教不会机械臂（205 个中伸臂 0-45，两个 seed 为 0）。
   - 名义起点数据里的触发靠时钟。
   - 输入加有限差分底盘速度（MLP 31）：伸臂 130/0/0，变成照抄。
   - GRU（记忆）把伸臂找回来一部分：149/101/16。
   - 底盘把机械臂当作阶段标志；停住之后的丢失，来自回归"零"指令残留的 6-10%（蠕动）。
   - 他的下一步是 E2（近目标起点）和带滞回的停止头。C3 不与之重复：C3 不重采数据、不加记忆，是"补触发状态"最便宜的做法。

## 3. 实验 C3（预注册）：触发帧加权

### 3.1 唯一变量：训练行

- 每个训练 episode（1613 个）除步长 9 的行之外，再加 K 份触发帧 k0（nav_done 的第一步）。
  - 训练缓存只有步长 9 的行，所以 k0 要从 data/train 的 1 kHz 录制读取。
  - 测量信号和标签都按录制原样：v = w = 0，力矩 = 专家开始伸臂那一步的力矩，不合成标签。
- 每份拷贝的机械臂 qd_arm 逐关节加 N(0, σ_hold,j)，其余输入不变。
  - σ_hold,j = 训练行中 nav_done 为假的行，机械臂 qd 的逐关节标准差。
  - 抖动用固定的 rng 只生成一次，所有训练 seed 共用同一组加进来的行。
- 以下全部不变：
  - 特征 ego_novw；
  - 标准化参数：只用原来的步长 9 训练行算，所以 K = 0 时与 ego-novw 完全相同；
  - 网络、超参数、划分；
  - 验证集与验证损失、early stopping。
- 实现：
  - train_policy.py 加 --trigger-copies K、--trigger-qd-jitter（默认按上面算 σ）。
  - 模型名 ego-novw-trig；meta 记录 K、σ_hold、抖动 seed、加进来的行数。
  - 校准用的模型命名为 trained/ego-novw-trig-k<K>_s0.npz；最终模型为 trained/ego-novw-trig_s<k>.npz。
- 单元测试（必须先通过）：
  - K = 0 时，在小子集上与 ego-novw 同一 seed 训练出的权重逐位相同；
  - 加进来的行数 = K x 训练 episode 数；
  - 拷贝的标签与录制逐位相同（v、w、7 个力矩）；
  - 只有 qd_arm 列被抖动，抖动可复现；
  - 标准化参数不变。

### 3.2 K 的确定（只用训练集和验证集，不看 DEV；在冻结预注册之后、训练正式 seed 之前做）

- 用 seed 0 训练 K ∈ {16, 64, 256} 三个模型，并行。
- 用 results/step3/C3_probe.py 计算。它复用 C0/probe_trigger.py 的帧集合和 r 的定义，在冻结前写好。
- 选满足下面两条的最小 K：
  - (i) EM 稳态 r 中位 >= 0.5；
  - (ii) 不误触发：验证集中环距离 > 0.3 m 的导航帧（nav_done 为假），r 中位 < 0.1，p95 < 0.3。
    这类帧的 r 用该 episode 的 k0-1 基线和 k0 跳变方向计算。
- 只报告，不作门槛：
  - PRE_LOOK 帧（在环附近、朝向、机械臂在保持姿态、nav_done 之前）的 r，即提前触发的倾向。触发帧与这些帧在单帧里分不开，预计会升高。
  - 导航段 (v, w) 的 nRMSE 和验证损失，与 ego-novw_s0 对比。
- 三个 K 都不满足：停下，向用户提方案。
- 选定的 K 训练出的 seed 0 模型应与校准模型 ego-novw-trig-k<K>_s0 逐位相同（检查确定性）。
- 然后训练 seed 0-4 和 100-104，共 10 个模型，并行。

### 3.3 运行（DEV）

- C3-EM（诊断）：
  - 命令：hybrid "base=expert;arm=trained/ego-novw-trig_s<k>.npz"，--no-early-stop，k = 0..4。
  - 输出：results/B-EM_ego-novw-trig_s<k>_dev。这个命名会被 parse_run_name 识别为 EM 格，并检查 policy_arg。
  - 对照：B-EM_ego-novw_s<k>_dev，也是 --no-early-stop。
- C3-MM（可部署、要判定的）：
  - 命令：bc_policy，trained/ego-novw-trig_s<k>.npz，早停（与 MM 相同），k = 0..4。
  - 输出：results/ego-novw-trig_s<k>_dev；加 --save-traj --traj-every 10（traj/ 已被 .gitignore 忽略），用于看触发时刻。
  - 按协议，把 stalled 的 episode 用 --no-early-stop 重跑到 results/step3/nostall/ego-novw-trig_s<k>_dev。除了 success，也报告决定指标有没有变化。
  - 敏感性（描述性）：两边都换成 stalled 重跑的行后，重算决定指标的差值。
- 分析命令：
  - analyze_rollouts.py compare --a ego-novw --b ego-novw-trig --set dev --seeds 0-4 --predicted-classes parked_no_arm,circling
  - C3-EM 对 B-EM 用 --a-runs/--b-runs。输出会固定写成 results/step3/A_A_vs_B_<set>.*，每次跑完立刻改名为 C3_EM_<set>.*，防止被下一次覆盖。
  - 机制预测写在 results/step3/C3_mechanisms.py 里（仿照 mechanism_checks.py，用显式目录），冻结前写好，输出 C3_mechanisms.txt。

### 3.4 竞争假设与预测（数字在运行前写死）

seed 平均；CI 同时用两层 bootstrap 和按 seed 的 t 区间，两者都要满足。

- **C3-EM 的判定**（指标 arm_extended_toward，DEV 300 个 episode；专家在同一批 episode 上 81.7%）：

  | 判定 | 条件 |
  |---|---|
  | 触发（H-support：缺的是"从静止开始伸"的样本） | 两个 CI 下界都 > 30% |
  | 不触发 | 两个 CI 上界都 < 20% |
  | 不确定 | 其余情况。停下报告，不做后续 |

- **H-trajectory**（触发了但够不准）：只在"触发"时判定。伸臂（arm_extended_toward）的 seed x episode 合计 >= 30 个时，看其中 10 mm 成功的比例；< 50% 即成立（专家 243/245）。失败类只在这些伸臂的单元里统计。
- **H-cooperation**（机械臂一伸，底盘就停；实验 B 的 ME 格）：C3-MM - MM 的 circling 差值，bootstrap 与 Welch 的 CI 上界都 < 0，并且 parked_no_arm 也下降。
- **H-premature**（加权把相像的帧也推向伸臂）：C3-MM 的伸臂中，ring_dist_at_arm_extend > 0.2 m 的比例 > 30%，主要失败类为 extended_not_reached 或 left_or_overshot。
  - 另报伸臂时刻相对影子 FSM nav_done 的时间：C3-EM 从 logs/ 读；专家自己的分布是 0.33-2.7 s，中位 0.47 s。
- **先验**：
  - C0 的推动对照显示，机械臂一旦开始动就会自我维持，所以只要触发帧学到了，伸臂很可能接着发生。
  - 但触发帧在单帧特征里与之前的 orient 帧分不开。所以学到的只能是"在环上、朝向 target 就伸"这条可观测规则，会比专家早触发（预测 H-premature 部分成立）。

### 3.5 主指标与地板规则

- 主指标：DEV success。
- 沿用第 3 步预注册的顺序，不在看过数值后另选：两边 seed 平均都 < 5% 时，决定指标改为 reach_5cm；两边 reach_5cm 也都 < 5% 时，改为 parked。
  - 局限：reach_5cm 在 MM 中主要来自不需要伸臂的短伸臂 episode（ego-novw 进入 5 cm 的 135 个单元中，84% 是这类）。所以同时报告 arm_extended_toward 作为机制证据。
- 决定指标只在 DEV 上确定，HARD、CONFIRM 沿用，不再重新应用地板规则。
- 决定指标不是 success 时，加一条"不变差"：C3-MM - MM 的 success 点估计 >= 0。

### 3.6 统计与 seed

- 统计同第 3 步：两层 bootstrap + 按 seed 的 Welch；分层和次要指标用 Holm。
- C3-MM 与 MM 是不同的网络，seed 不配对；按 k 的配对 t 只作描述。
- seed 固定为 0-4（CONFIRM 用 100-104）。看过 DEV 结果之后：不再加 seed，不换分析方法。
- 如果 bootstrap 满足而 Welch 不满足：报告每个 seed 的差值和 Cochran Q，写成"效应只在部分 seed 上出现"，不采纳。

### 3.7 采纳规则（C3-MM 替代 MM 成为主线）

全部满足才采纳：
- (1) DEV 上决定指标的差值（C3-MM - MM）：bootstrap 与 Welch 的 95% CI 下界都 > 0。
- (2) 改进来自预测的机制：决定指标净改进中 >= 50% 来自 parked_no_arm + circling 的减少，即 compare 的 "Share of the net difference from ..."。
  - 决定指标不是 success 时，在该指标失败的 episode 里统计，算法同实验 A，见 report.txt。
  - 另外报告"在环上朝向时伸臂"的比例：arm_extended_toward 且 ring_dist_at_arm_extend < 0.20 m 且 |target_bearing_err_at_arm_extend_deg| < 6；专家 DEV 236/245、HARD 159/181。
  - 不用 parked_time 判断"先停后伸"：它是"静止 1 s"区间的结束时刻，专家 245 个伸臂中只有 8 个满足。
- (3) n >= 30 的 d0 分层中，没有一层的差值 CI 上界低于 -10 个百分点。
- (4) HARD：C3-MM 5 次，与 results/ego-novw_s<k>_hard 比较，方向一致。
- (5) CONFIRM：trained/ego-novw-trig_s10<k>.npz，与 results/ego-novw_s10<k>_confirm 比较，方向一致。

反方向（MM 明显更好，(1)-(4) 镜像成立）：记为"触发帧加权有害"的证据。
都不满足：保留 MM，写成"在约 X 个百分点的分辨率下没有检测到差异"。

### 3.8 冻结

- 实验 C3 的预注册单独成一节："[Methodology] 第 3 步续预注册：实验 C3（日期，在 DEV/HARD/CONFIRM 上运行 C3 之前）"。
- sha256 只算这一节（到下一个以 "[" 开头的标题为止），照 results/step3/prereg_frozen.txt 的格式记在 results/step3/prereg_C3_frozen.txt。
- 冻结的范围也包括：train_policy.py 的改动、C3_probe.py、C3_mechanisms.py。

## 4. 决定表

| 情况 | 下一步 |
|---|---|
| K 校准不过 | 停下提方案。候选：触发分类器门控（nav_done 标签、类别加权）；机械臂 qd 照抄的处理；Rodrigo 的 GRU 路线（这里没有 torch，需与他协调） |
| C3-EM 不确定 | 停下报告 |
| C3-EM 不触发 | 停下提方案（同上；并报告探针 r 与闭环的差异） |
| C3-EM 触发且 H-trajectory 成立 | 先照常判定 C3-MM；然后报告触发与轨迹两部分的分解，停下提方案（例如机械臂单独建模、按 z 分组） |
| C3-EM 触发、C3-MM 满足 (1)-(3) | 跑 HARD、CONFIRM，按采纳规则决定 |
| C3-EM 触发、C3-MM 不满足 | 报告原因（H-premature、绕圈或 seed 异质），停下提方案 |

## 5. 步骤与预算

1. 读材料。git fetch team，看 Rodrigo 的新提交。复跑 C0 的脚本，确认输出与 results/step3/C0/*.txt 一致。约 30 分钟。
2. 实现 train_policy.py 的 --trigger-copies 及单元测试；写 C3_probe.py、C3_mechanisms.py。全部单元测试通过。约 1.5 小时。
3. 写预注册，把第 2 节的依据也写进 report.txt，然后冻结。约 30 分钟。
4. K 校准：3 个模型并行，加探针。约 40 分钟。
5. 训练 seed 0-4 和 100-104：10 个并行。约 40 分钟（实验 A 时 10 个并行约 30 分钟）。
6. DEV：C3-EM 5 次（约 1 小时）+ C3-MM 5 次（约 1 小时）+ stalled 重跑；分析，按第 4 节分支。
7. 若 (1)-(3) 满足：HARD 5 次（约 40 分钟）、CONFIRM 5 次（约 1 小时）。
8. 结果写进 report.txt。
   - 运行 analyze_rollouts.py report --no-figures 更新 summary.txt，用 git diff --stat 核对 A_*、B_* 只有时间戳变化。
   - 加图 results/step3/fig_step3_C3_arm.{png,pdf}：B-EM、C3-EM、MM、C3-MM、专家各格的 success、arm_extended_toward，seed 平均加两层 bootstrap CI。
   - 告诉用户结论和下一步。只在用户要求时提交。
9. 与第 4 步的关系：学习曲线、降采样步长放在 C3 之后，在 C3 采纳的模型上做。若 C3 没有通过，第 4 步只报告离线指标和探针 r 随数据量 / 步长的变化（描述性），不跑闭环。

总计约 7-8 小时，大部分可以无人值守。

## 6. 产出与局限

- 代码：train_policy.py（--trigger-copies）、tests/test_training.py 的新测试、results/step3/C3_probe.py、C3_mechanisms.py、run_expC3.sh。
- 结果：results/B-EM_ego-novw-trig_s<k>_dev、results/ego-novw-trig_s<k>_{dev,hard}、results/ego-novw-trig_s10<k>_confirm、results/step3/C3_*.txt、图。
- report.txt：实验 C3 的预注册（运行前）、K 校准、结果、局限。局限包括：
  - (1) C3 只改变训练行的权重：加进来的 k0 帧是专家真实访问过、真实标注的状态，没有合成标签；只给 qd 加了抖动。
    它针对的是"学习者停在专家几乎不经过的状态（底盘已停、机械臂未动）"这一协变量偏移，属于 DAgger（Ross, Gordon, Bagnell, AISTATS 2011）、DART（Laskey et al., CoRL 2017）一类的问题，但没有向专家查询新状态。
    引用前核对出处，写进 report.txt 的 [References]。
  - (2) C3-EM 的专家底盘给了完美的停靠，只作诊断。
  - (3) 探针 r 是离线的必要条件，不保证闭环触发；闭环结论只来自 DEV/HARD/CONFIRM。
  - (4) K 是在验证集上选的，不是在 DEV 上选的。
  - (5) 学到的触发是可观测规则，可能比专家早，见 H-premature。
