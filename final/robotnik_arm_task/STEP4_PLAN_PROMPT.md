# 第 4 步方案 prompt：实验 D（末端误差特征，预注册）与 E（学习曲线 / 降采样步长，描述性），并行

2026-10-31 定稿，过程如下：
- 实验 C3 被采纳后，用户选了"末端误差特征做预注册实验（实验 D），第 4 步 E 并行"。
- 初稿经两个独立审查（对照代码的可行性；统计设计与价值），共 33 条意见。
- 用户随后确认了 4 个决定：
  - D 每边 10 个 seed；
  - 正运动学用完整底盘姿态；
  - 误差特征用近场截断版；
  - E 先跑缩减版，结果没有参考意义再考虑扩展。

本文件交给执行的会话。按用户的工作方式：
- 先读材料、实现代码与测试；把预注册写进 report.txt 并冻结，然后才运行。
- 遇到方案没有覆盖的情况，或者第 6 节写着"停下"，就停下来向用户提方案，等确认后再继续。

## 0. 背景与规则

- 目录、Python、评估集、统计方法、提交规则：同 STEP3C_PLAN_PROMPT.md 第 0 节。
  - 任务目录 final/robotnik_arm_task，本地分支 worktree-eval-protocol。
  - pixi Python，unittest。
  - DEV 做所有决定，CONFIRM 只在采纳前复现一次，test 封存。
- team：C3 已推为 feature/step3c-trigger-copies（cd92341），叠在 feature/step3-bc 之上。新工作推成新的 feature/<task>，
  基于 feature/step3c-trigger-copies cherry-pick。只在用户要求时提交 / 推送。
- 截止 2026-11-06 16:00（报告最多 6 页）。目标：11-01 前 D、E 出结果，之后写报告。
- 分工：到达后保持不住（底盘蠕动）是 Rodrigo 的方向（team/rodrigo 的 docs/BLACKBOX_PLAN_20261027.md §6h-§6j），
  这里不做，只在报告中引用。开始前 git fetch team，看 7dc1345 之后的提交。
- 开始前先读：
  - report.txt 的"[Results] 第 3 步续：实验 C3 结果"全部（特别是第 5、6、8、10 条）；
  - 第 2 步结果中低 z 抖动一段（"抖动：z < 0.45 时 ..."）；
  - 第 2 步对第 4 步的事先说明（"第 4 步降采样步长扫描：加'窗口内平均后再抽取'作为对照"）。
  - 代码：train_policy.py、bc_features.py、policies/bc_policy.py、policies/hybrid.py、evaluate_policy.py
    （setup_simulation 的 info、ExpertPolicy.act 与 harness 的正运动学）、collect_data.py（ee_pos 的录制时机）、
    robotnik_arm_control.py（arm_controller 的 ik_error）、analyze_rollouts.py、results/step3/C3_mechanisms.py、
    run_expC3.sh、EVALUATION.md（L66-69 停滞早停检查；L146-147 使用 info 的规定）。

## 1. 现状与事后诊断（预注册之前已看到，写进 report.txt）

- 主线 ego-novw-trig（K = 16）：success DEV 60.5%、HARD 56.3%、CONFIRM 64.3%（专家 97-99%）。
- DEV 失败（seed 平均）：extended_not_reached（ENR）18.7%、circling 12.1%、侧翻 3.2%、approached_other 3.4%、
  left_or_overshot 2.1%；held_10s 26.9%。
- C3 结果第 10 条：
  - 低处 target（z < 0.45）success 20.3%，高处 72.5%；专家底盘下低处也只有 17.7%。
  - 侧翻全部发生在伸臂之后。
  - ENR 的最小误差中位 16 mm，51% 停在离 target 1-4 cm 处不动。
- 审查时又看到的（也写进预注册的"已看到"）：
  - C3-MM 中"失败、朝 target 伸臂、最小误差在 [1, 5) cm"的单元 458 个：ENR 271，circling 125，crash 35，
    left_or_overshot 27（类别有优先级，约 40% 精度类的失败不叫 ENR）；底盘已停靠的 362 个中 ENR 248。
  - ENR 中 62.5% 的最小误差已经 < 2 cm（reach_2cm 对这个机制不敏感）。
  - 训练缓存的 nav_done 行中，末端误差 10-20 mm 且机械臂静止（max|qd_arm| < 0.02 rad/s）的只有 0.6%（低处）/ 1.3%
    （高处），20-40 mm 时为 0%：学习者停住的状态，专家数据里几乎没有。
  - 正运动学复现录制 ee_pos：21 个 episode、6,264 步上最大差 3.9e-7 m；每步约 4 us。
  - 底盘倾斜与下沉对末端位置的影响：中位 4.9 mm（最大 9.8 mm）。
  - 低 z 抖动严格以 2 为周期（98.5% 的相邻抖动步异号）；直接抽取的偏差：步长 9 为 0.05 N·m，步长 8 为 0.73 N·m。
- 统计功效（C3-MM，DEV success 每个 seed 的标准差 6.7 pp）：
  - 每边 5 个 seed：Welch 半宽约 9.7 pp，80% 功效的可检测差异约 14.5 pp；
  - 每边 10 个 seed：约 9 pp。
  - 这个机制最多大概带来约 10 pp（静态精度误差约占 ENR 的一半）。

## 2. 实验 D（预注册）：末端误差特征

### 2.1 唯一变量：输入特征

- 特征集 ego_novw_ee（23 维）= ego_novw（20 维）+ 3 个近场截断的末端误差分量。模型名 ego-novw-trig-ee。
  - e = target - ee_pos（世界坐标，ee_pos 是完整底盘位姿下的正运动学）；
  - e_c = e · min(1, 0.10 / |e|)：长度截在 10 cm，方向不变；
  - (ee_dx, ee_dy) = e_c 的水平分量旋转到当前底盘 yaw 坐标系（同 bc_features.ego_target），ee_dz = e_c 的 z 分量。
  - 理由（写进 report）：训练行 |e| 均值 0.72 m、标准差 0.69 m，不截断时 16 mm 的误差只有 0.02-0.05 个标准差；
    截断后 1-2 cm 约 0.3-0.5 个标准差。远处的信息已由 ego_dx、ego_dy、target_z、rho 给出。
- 其余全部与 ego-novw-trig 相同：
  - 训练行、划分、步长 9；
  - K = 16 的触发帧拷贝，同样的 σ_hold 与抖动 seed；拷贝的 e_c 用录制的 k0 步 ee_pos（抖动只改 qd）；
  - 标准化只用原来的训练行（新列也一样）；
  - 网络、超参数、early stopping、验证集。
- 注意：输入维度变了，初始化与 early stopping 的轨迹也随之改变（同 C3 结果第 5 条的混杂）；归因写成
  "末端误差特征及其对训练的影响"。
- 末端位置的来源：
  - 训练：录制的 ee_pos（专家每一步用 pinocchio 算，float32，与 base_pos、base_ori、q_mes 同一步，在 sim.Step 之前）。
  - 部署：BCPolicy 只对需要末端的特征集，在 reset 时从 info 取 dyn_model、ee_frame_id；每一步做与专家相同的调用：
    ReoderJoints2PinVec(base_pos, base_ori, q_mes) -> forwardKinematics -> updateFramePlacements -> oMf[ee].translation。
    hybrid 的 EM 格里，BCPolicy.reset 同样拿到 info，hybrid.py 不需要改。
  - 旧特征集的 BCPolicy 行为完全不变，reset(None, None) 仍然可用。
- 设计说明（写进 report，并改写 bc_policy.py 的说明与 report 第 3 步"全部是当前观测的纯变换"那一句）：
  - 这是 policy 第一次用 info 里的机器人运动学模型。EVALUATION.md L146-147 允许，但要在报告中说明。
  - 末端位置是当前观测（q、底盘位姿，含 obs 里的 base_ori）的确定函数，不用停靠点、规划、阶段或时间。
  - 但它恰好是专家 IK 的误差输入（robotnik_arm_control.py 的 ik_error 在臂展内等于 target - ee_pos），也是成功判据测量的量。
    即"用已知模型做特征工程，把正运动学从学习问题里拿掉"，对应 brief 的"额外的传感器 / 输入数据有没有帮助"。
  - 若 D 被采纳，主线不再是"只用测量信号"；报告中同时给出 C3 模型作为"不用机器人模型"的结果。
- 实现（D 的代码只在下面这些文件里；E 的代码放在单独的 train_step4e.py，D 冻结后 train_policy.py 到 CONFIRM 之前不再改）：
  - 旁路缓存 results/step3/cache/ee/{train,val,trigger}/ee_pos.npy：与现有缓存逐行对齐的录制 ee_pos（float32）。
    manifest 记录它所对应的主缓存与触发缓存 manifest 的 sha256。现有缓存与 C3 的冻结记录都不动。
  - bc_features.py：特征集 ego_novw_ee 与截断函数；measured_signals 增加可选的 ee_pos。
  - train_policy.py：模型 ego-novw-trig-ee；训练时从旁路缓存读 ee_pos。
  - policies/bc_policy.py：正运动学；没有 dyn_model 时，ego_novw_ee 模型拒绝运行（不悄悄用 0）。
- 检查：
  - 用新代码重新训练 ego-novw-trig_s0（--out results/tmp_retrain/...），W*、b*、x_*、y_* 共 10 个数组与
    trained/ego-novw-trig_s0.npz 逐位相同（meta 只有时间与代码 hash 不同）。
  - 回归测试：用 git show 取出 HEAD 的 bc_features.py 与 bc_policy.py，对 trained/ego-novw-trig_s0.npz 在 300 个录制步上，
    新旧 BCPolicy.act 逐位相同。可选：DEV --limit 20 到 results/tmp_*，与 results/ego-novw-trig_s0_dev 对应行逐列相同。
- 单元测试（必须先通过）：
  - 部署时的正运动学复现录制的 ee_pos：若干录制 episode（含带噪的、离原点远的）的 300 步上最大差 < 1e-6 m。
  - 旁路缓存的 ee_pos 与录制逐位相同，行与主缓存对齐（episode、step 列相同）。
  - 截断：|e| < 0.10 时不变，否则长度为 0.10、方向不变；特征在底盘与 target 的平面旋转 + 平移下不变（< 1e-9）。
  - 训练 / 部署一致，分两层：(i) 特征层，末端 3 列 < 1e-6（float32 录制的舍入），其余列 < 1e-9；
    (ii) 输出层，向量化路径喂入部署时重算的 ee_pos，与 BCPolicy.act 一致 < 1e-9。
  - 旧特征集：特征与旧缓存逐位相同；reset(None, None) 可用；无 dyn_model 时 ee 模型拒绝。
  - policy_ms 报告（正运动学每步约 4 us）。

### 2.2 seed 与模型

- DEV、HARD：每边 10 个 seed（0-9）。
  - D：新训练 ego-novw-trig-ee_s0-9。
  - C3 对照：已有 s0-4，补训 ego-novw-trig_s5-9（冻结的 C3 配方，K 16，代码路径不变）。
- CONFIRM：每边 seed 100-104。D 新训练 ego-novw-trig-ee_s100-104，C3 用已有的。
- 训练检查（照 run_expC3.sh train）：每个 D 模型的 meta 是 K 16、自动 σ、抖动 seed 0、特征集 ego_novw_ee，
  train_policy.py 与 bc_features.py 的 sha256 等于冻结值；补训的 C3 s5-9 同样核对 K 与代码 hash。

### 2.3 运行（全部新运行用 --no-early-stop）

- 理由：C3 这一族没通过停滞早停检查（DEV 96 个 stalled 中 5 个不早停会成功），按 EVALUATION.md L66-69，这族模型
  不能用早停。不早停的运行时长与早停几乎一样。
- 已有 C3 运行的处理：早停的运行中，没有被判 stalled 的 episode 与不早停时逐步相同（仿真确定），所以对照组 =
  已有运行 + stalled episode 的 --no-early-stop 重跑行。
  - DEV s0-4 的重跑已有（results/step3/nostall/ego-novw-trig_s<k>_dev）；
  - HARD s0-4、CONFIRM s100-104 的 stalled 要补跑到 results/step3/nostall/。
- D-EM（诊断）：hybrid "base=expert;arm=trained/ego-novw-trig-ee_s<k>.npz"，k = 0..4 -> results/B-EM_ego-novw-trig-ee_s<k>_dev；
  对照已有的 C3-EM（B-EM_ego-novw-trig_s<k>_dev，本来就是 --no-early-stop）。
- D-MM：bc_policy，DEV k = 0..9（--save-traj --traj-every 10）、HARD k = 0..9、CONFIRM k = 100..104。
- C3-MM 补跑：DEV、HARD 的 k = 5..9。
- 分析：
  - analyze_rollouts.py compare：C3 一边用"已有运行 + 重跑行"合成的行（在 D_mechanisms.py 里做，或把合成后的
    rollouts 写到 results/step3/C3merged/），--predicted-classes extended_not_reached。
  - 全部判定由 results/step3/D_mechanisms.py 计算（仿照 C3_mechanisms.py），冻结前写好，并在已有的 C3 运行上干跑
    （拿 C3 自己当替身）。

### 2.4 竞争假设与预测（数字在运行前写死；seed 平均；两层 bootstrap 与 Welch 都要满足，另有说明的除外）

- 先决检查：触发不变。D-EM 与 C3-EM 的 arm_extended_toward 差值，两个 CI 的上界都 >= -5 pp，并且每个 D-EM seed >= 70%
  （C3-EM 每个 seed 78.7-80.0%）。不满足：停下报告"D 改变了触发"，D-MM 只作描述。
- 定义：
  - "需要伸臂的 episode"：专家参照 navdone_ee_err >= 0.07 m（DEV 262 个），两边相同，不以处理后的变量为条件。
  - "精度池"：失败、朝 target 伸臂、底盘已停靠（parked_time 非空）、最小误差在 [1, 5) cm，不论失败类别。
  - "细精度指标"：需要伸臂的 episode 中最小误差 < 15 mm 的比例（C3-MM 每个 seed 76.7 / 63.7 / 74.0 / 70.6 / 64.5%）。
- H-precision（静态误差来自网络要从关节角隐式学正运动学）。预测：
  - D-MM - C3-MM：精度池比例的两个 CI 上界都 < 0；细精度指标的两个 CI 下界都 > 0。
  - D-EM - C3-EM（需要伸臂的 262 个 episode 上，无条件的 success、精度池比例、细精度指标的配对差值）：描述性，
    只报估计与 CI（EM 的 seed 标准差约 9.7 pp，5 个 seed 功效低）。
- H-label（低处的精度被 bang-bang 标签卡住，与输入无关）。交互检验：
  - Δ_high（z >= 0.45 的 success 差值）与 Δ_low（z < 0.45），以及 Δ_high - Δ_low 的 CI（两层 bootstrap + Welch）。
  - H-label 成立 = Δ_high 的两个 CI 下界都 > 0，并且 Δ_low 的两个 CI 都在 ±5 pp 内（低处基线 20%；按低处 success 的
    seed 标准差 3.8 pp、每边 10 个 seed，Welch 半宽约 3.6 pp）；Δ_high 不显著时记为"不可判定"。
- H-coverage（不利；学习者停住的状态在专家数据里几乎没有，加特征不增加这类样本）。预测：精度池的减少 < 3 pp。
  - 离线探针（只报告，冻结前写好，results/step3/D_probe.py）：
    - 验证集 reach 段（nav_done 之后、首次进入 10 mm 之前）末端误差 10-40 mm 的帧，令 qd_arm = 0；
    - 每个模型的预测力矩减去重力矩 g(q)（pinocchio，录制的底盘位姿）；
    - 投到能减小 e 的关节方向 J^+ e 上（J 为末端平移雅可比）；
    - 报告"从静止推向 target"的帧比例与投影大小，C3 对 D（seed 0-4）。
- H-shortcut（不利；e_c 也是"离 target 多远"的信号，模型可能更早伸臂）。预测：
  - D-MM 伸臂时环距离 > 0.2 m 的比例：合计 > 10%，至少 3 个 seed > 5%，伸臂单元 >= 30（C3-MM 合计 2.0%）；
  - circling + left_or_overshot 差值的两个 CI 下界都 > 0 记为"底盘受影响"。
- 侧翻（描述性）：精度提高可能减少侧翻，也可能因为伸得更深而增加。
- 先验：H-precision 在高处成立，低处受 H-label 或 H-coverage 限制，净改进主要来自 z >= 0.45，幅度 5-10 pp。

### 2.5 主指标与采纳规则

- 决定指标：DEV success（地板规则照旧，按构造不会生效）。
- 采纳（全部满足才把主线从 ego-novw-trig 换成 ego-novw-trig-ee）：
  - (0) 触发不变（2.4 的先决检查）；
  - (1) DEV success 差值（D-MM - C3-MM，10 vs 10 个 seed）：bootstrap 与 Welch 的 95% CI 下界都 > 0；
  - (2) 改进来自预测的机制：[C3 中精度池比例 - D 中精度池比例] / 净改进 >= 50%（只 ENR 的份额另报）；
  - (3) n >= 30 的 d0 分层，没有一层的 CI 上界低于 -10 个百分点（与 C3 相同，只用 bootstrap 的分层 CI）；
  - (4) HARD（10 vs 10）点估计 > 0；
  - (5) CONFIRM（5 vs 5）点估计 > 0，只在 (0)-(4) 满足后打开一次。
  - 另报（Holm）：held_10s、侧翻、reach_2cm、arm_extended_toward、细精度指标；z 分层。
- 敏感性（预注册的报告项）：C3 一边改用原来的早停行（不换成重跑行）重算 (1)、(2)，报告与主分析的差别；结论不同时如实
  报告，以不早停的主分析为准（协议规定）。
- statistics 细则（同 C3）：
  - bootstrap 满足而 Welch 不满足：报告每个 seed 的差值与 Cochran Q，写成"效应只在部分 seed 上出现"，不采纳；
  - C3 与 D 是不同的网络，seed 不配对，按 k 的配对 t 只作描述；
  - 看过 DEV 之后不再加 seed、不换分析方法。
- 反方向（C3 明显更好）：(1)-(4) 镜像成立，记为"末端误差特征有害"的证据；DEV 上镜像 (1)-(3) 成立时停下提方案，HARD 作为
  提议的一部分。
- 都不满足：保留 ego-novw-trig，写成"在约 X 个百分点的分辨率下没有检测到差异"（X = Welch 与 bootstrap CI 半宽中较大的）。

### 2.6 冻结

- 预注册单独成一节："[Methodology] 第 4 步预注册：实验 D（日期，在 DEV/HARD/CONFIRM 上运行 D 之前）"。
- 节内第 0) 条"预注册之前已经看过的东西"：第 1 节全部、D_mechanisms.py 的干跑、单元测试的小模型、离线探针在 C3 模型上的值
  （不看 D 模型的），以及"没有看过：任何 ego-novw-trig-ee 模型（全训练集）的损失、探针或闭环结果"。
- sha256 只算这一节，记在 results/step3/prereg_D_frozen.txt。冻结记录列出以下文件的 sha256：
  train_policy.py、bc_features.py、policies/bc_policy.py、policies/hybrid.py、evaluate_policy.py、analyze_rollouts.py、
  tests/*.py、D_mechanisms.py、D_probe.py、run_expD.sh、trained/data_split.json、旁路缓存 manifest。
- E 的事先读法单独成一节 [Methodology]，放在 D 那一节之后（改 E 的文字不影响 D 的 hash）。

## 3. E：第 4 步（描述性，缩减版；在 C3 的配方上；不作采纳决定）

- 目的：数据量与时间密度对离线误差和闭环表现的影响，给报告的分析部分。
- 在 C3 配方上做，与 D 并行；即使 D 被采纳，E 也不在 D 上重做（如实写明）。
- 用户的决定：先跑缩减版，结果没有参考意义再考虑扩展（步长 3、窗口平均对照等）。

### 3.1 设置（每个设置 seed 0-2）

- 学习曲线：训练 episode 比例 f ∈ {1/8, 1/4, 1/2}；f = 1 用已有的 ego-novw-trig_s0-2。
  - 嵌套分层子集（noisy x mode，每层 round(f·n)，每层一个固定排列 default_rng(1) 取前缀）：1/8 = 201、1/4 = 404、
    1/2 = 807 个 episode，写成 trained/data_subsets.json。
  - 子集的 episode 各加 16 份触发帧；偏移、步长 9、验证集不变；标准化与 σ_hold 都用子集自己的训练行（记进 meta）。
- 降采样步长：s ∈ {27, 81}；s = 9 用已有的。都是奇数。
  - 每个步长一个划分变体 trained/data_split_s<s>.json：训练 / 验证 episode 不变，偏移 default_rng(0).integers(0, s)。
  - 自己的缓存 results/step3/cache_s<s>/ 与触发缓存 results/step3/cache_s<s>/trigger。检查：触发缓存各列与 stride 9 的
    逐位相同（k0 来自 1 kHz 录制，与步长无关）。
  - 拷贝按步长缩放，保持约 1.4-1.7% 的行占比：K = round(16 x 9 / s) = 5 / 2（舍入的偏差写明）。
- 模型名（每个变体独立，避免与主线撞名）：ego-novw-trig-f0125 / -f025 / -f05 / -s27 / -s81。
  - "名字 -> 数据规格"的表（划分文件、缓存、触发缓存、K、子集比例）在 train_step4e.py 里；命令行参数不一致时拒绝。
  - meta 记录划分文件与 data_subsets.json 的 sha256。不用 --force。
- 优化步数固定：E 的每次验证之间走约 1.79M 个样本（一个 f = 1、s = 9 的 epoch），patience 5 次检查，最多 200 次检查。
  - f = 1/8、1/4、1/2：每次检查之间调用 8 / 4 / 2 次 partial_fit；
  - s = 27、81：每次检查之间调用 3 / 9 次 partial_fit。
  - meta 记录总样本数、最佳检查点、stop_reason。
  - 避免"数据量"与"优化步数"混在一起（审查意见；C3 结果第 5 条的同类混杂）。
- 第 2 步预注册的抗混叠对照：改用数据层证据回答，不训练（奇数步长的抽取偏差 0.05 N·m 对偶数 0.73 N·m，低 z 抖动严格以
  2 为周期）。在 report 里写明偏离第 2 步的原计划及原因。
- 新模型 15 个；闭环 DEV 15 次，全部 --no-early-stop（与 D 同样的理由）。

### 3.2 报告与事先读法（不作判定；读法写进 E 那一节，运行前写好）

- 每个新模型：
  - 离线：验证损失；导航段 (v, w) nRMSE；力矩 nRMSE 按段 x z（对原始 1 kHz 验证标签）。
  - 闭环：success、细精度指标、arm_extended_toward、held_10s、失败类。
  - 每个 seed 的点附 Wilson 区间；每个设置的 seed 平均附两层 bootstrap CI。3 个 seed 的 CI 偏窄，两个设置之差的
    95% 半宽约 ±15 pp，如实写明。
- 图：
  - results/step3/fig_step4_learning_curve.{png,pdf}：横轴训练 episode 数（对数），纵轴 success 与离线验证损失，画出每个 seed。
  - results/step3/fig_step4_stride.{png,pdf}：横轴训练行数（对数），同上。
  - 两张图都标出行数相同的一对：f = 1/8（约 0.22M 行，201 个 episode）对 s = 81（约 0.20M 行，1613 个 episode）。
- 读法：
  - P-E1：success 对 log2(训练 episode 数) 的斜率（12 个模型，按 seed 的残差，t 区间）> 0；"接近饱和"写成
    1/2 -> 1 的增量 CI 上界 < 5 pp（3 个 seed 下大概率不可判定，照写）。
  - P-E2：行数相同时，episode 多的（s = 81）比 episode 少的（f = 1/8）好：差值的估计与 CI（多样性比行数重要）。
  - P-E3：s = 27、81 比 s = 9 差多少（估计与 CI）；验证损失随行数的变化。
  - P-E4：离线验证损失与闭环 success 的关系仍然很弱或反向。
- 跑完后与用户讨论是否扩展（步长 3、力矩窗口平均对照、更多 seed）。

## 4. 实现与运行顺序（计算预算）

1. 读材料，git fetch team。约 20 分钟。
2. D 的代码、测试、D_mechanisms.py、D_probe.py、run_expD.sh；在已有 C3 运行上干跑；探针在 C3 模型上的值。约 3-4 小时。
3. 写 D 的预注册，冻结；写 E 的事先读法。约 30 分钟。
4. E 的代码（train_step4e.py、run_step4E.sh、作图脚本）可以在 D 冻结之后写，不碰 D 冻结的文件。约 2 小时。
5. 训练（不与评估重叠；每批 <= 12 个进程；开始前查 free）：
   - D：20 个模型（ee s0-9、s100-104，C3 s5-9）+ 1 个重训检查，约 1 小时；
   - E：15 个，约 1 小时。
6. 评估队列（一次一个运行，20 个 worker）：
   - D：EM 5、DEV MM 15、HARD 15、C3 的 HARD / CONFIRM stalled 补跑；按决定表 CONFIRM 5。约 5-6 小时。
   - E：DEV 15 次，约 2.5-3 小时，可以过夜。
7. 结果写进 report.txt，图，summary.txt（用实验 A 的参数：--allow-confirm report --no-figures --predicted-classes
   no_start,no_progress,left_or_overshot，核对 A_*、B_* 不变）。告诉用户结论。只在用户要求时提交。

## 5. 产出

- 代码：bc_features.py（ego_novw_ee）、train_policy.py（ego-novw-trig-ee、ee 旁路缓存）、policies/bc_policy.py（正运动学）、
  tests；train_step4e.py；results/step3/D_mechanisms.py、D_probe.py、run_expD.sh、run_step4E.sh、作图脚本。
- 结果：D 的 EM / DEV / HARD / CONFIRM 运行、C3 的补跑与重跑、E 的 15 个 DEV 运行、results/step3/D_*.txt、step4_*.txt、图。

## 6. 决定表

| 情况 | 下一步 |
|---|---|
| 重训的 ego-novw-trig_s0 数组不同，或旧模型的部署输出变了 | 停下查原因 |
| 正运动学复现不了录制的 ee_pos（> 1e-6 m） | 停下提方案 |
| D 的先决检查不过（触发变了） | 停下报告，D-MM 只作描述 |
| D-MM 满足 (1)-(3)，H-shortcut 不成立 | 跑 HARD；(4) 满足再跑 CONFIRM；按采纳规则决定 |
| D-MM 满足 (1)-(3)，但 H-shortcut 也成立 | 先照常判定，然后停下与用户讨论（改进可能来自更早伸臂），HARD、CONFIRM 作为提议的一部分 |
| bootstrap 满足、Welch 不满足 | 报告"效应只在部分 seed 上出现"，不采纳，停下提方案 |
| D-MM 不满足、H-label 成立 | 报告分解，停下提方案（例如低处的力矩标签平滑或按 z 加权） |
| D-MM 不满足、H-coverage 成立 | 报告，停下提方案（例如 reach 段的 DART 式扰动数据） |
| DEV 上镜像 (1)-(3) 成立 | 停下提方案，HARD 作为提议的一部分 |
| HARD (4) 不满足 | 不采纳，不打开 CONFIRM |
| 其余不满足 | 报告原因，保留 ego-novw-trig |
| 任何运行出现 harness_error | 重跑那个运行 |

几行重叠时，"停下"的那一行优先。

## 7. 局限（写进 report）

- (1) policy 用了机器人的运动学模型（正运动学），这是对"只用测量信号"的一个扩展；误差特征恰好是专家 IK 的输入和成功判据的量。
- (2) D 用完整底盘姿态（含倾斜，约 3-5 mm 的影响），所以 D = 显式正运动学 + 倾斜信息，不只是"不用学正运动学"。
- (3) D 只改输入不改标签：低处的 bang-bang 标签、学习者停住状态的数据覆盖都可能仍然限制精度。
- (4) E 在 C3 的配方上做、每个设置 3 个 seed，是描述性的；步长扫描中拷贝数随步长缩放（K 48 / 16 / 5 / 2 的规则，
  实际跑 16 / 5 / 2）；E 的抗混叠问题用数据层证据回答，没有训练窗口平均的模型。
- (5) 到达后保持不住不在本步范围内（Rodrigo 的方向）。
