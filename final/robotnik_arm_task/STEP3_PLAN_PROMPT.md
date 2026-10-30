# 第 3 步方案 prompt：第一批学习模型的闭环评估

（2026-10-29 修订。经过 3 个独立审查：统计设计、可行性实测、对团队和 report 的价值。
第 10 节的决定用户已于 2026-10-29 确认，可以按本方案执行；预注册内容仍须在运行前写进 report.txt。）

## 0. 背景与规则

- 工作目录：/home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task
  - 当前分支：worktree-eval-protocol。
  - 小组仓库是 remote team，分支约定 feature/*。上一次推送是把提交 cherry-pick 到 team/main 上，作为 feature/eval-protocol 推上去的。
- 开始前先读：
  - report.txt：全部设计决定、协议 v1、第 2 步的预注册与结果，中文。
  - EVALUATION.md、evaluate_policy.py：policy 接口是 make_policy(arg)，返回一个有 reset(target, info) 和 act(obs) 的对象。
  - DATA_COLLECTION.md、collect_data.py、analyze_dataset.py。
  - results/data_analysis/summary.txt、episodes_{train,dev,hard}.csv：有每个 episode 的 d0、mode、align_req、navdone_ee_err、覆盖度百分位。
- 队友 Rodrigo 的分支 team/rodrigo：用 git show team/rodrigo:<path> 读，不要 checkout。
  重点文档：docs/MECHANISMS_20261027.md、docs/BLACKBOX_PLAN_20261027.md（§6-§6e）、docs/PRIVILEGED_ABLATION.md、docs/NIGHT_RUN_20261027.md。
- Python：pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python ...
  - 有 numpy、scipy、sklearn 1.9.1、pinocchio、pybullet；没有 torch、pandas。
  - 机器：24 线程（8P + 16E）、30 GiB 内存。
- 数据：
  - data/train 2000 个，过滤后（reached 且没有导航超时）1897 个。
  - data/pilot：seed 1000，100 个，独立，只用于校准和冒烟测试。
  - DEV（seed 200，300 个）：所有决定都在这里做。
  - HARD（seed 201，200 个）：短距离 drive。
  - CONFIRM（seed 202）：采纳的改动在这里复现一次。
  - test（seed 100）：封存，不碰。
- 工作方式：
  - 每个实验先在 report.txt 写好假设、竞争假设、机制预测、主指标和最小可检测差异、采纳规则，再运行；一次只改一个变量。
  - 所有比较都在同一批 episode 上配对；多 seed；给出 CI。不凭空推测，也不因为"试出来更好"就采纳。
  - 预注册之前看过的东西都要如实记录，包括 pilot 冒烟测试。值得写进 report 的内容追加到 report.txt 并告诉用户。只在用户要求时 commit。

## 1. 来自队友的先验（单独列出，不算我们的复现）

- Rodrigo 的设定与我们的差别：
  - 专家的接近方向从世界原点算；我们的从起始位置算，两者只在原点起步时相同。
  - 他大多是原点起步；我们全部是随机起点。
  - 他的 21 维输入用原始 yaw，没有轮速；有 clock（t/40）。
  - DART 加噪 100% 的 episode，参数比我们的大，到达后也继续加噪；保留专家失败的 episode。
  - 评估固定 40 s，以 1/2/5 cm 内停留 0.5 s 判成功，头条用 5 cm；轮速限幅到 ±5.27。
- 他实测的结果（他自己 205 个原点起步的验证目标）：
  - 21 维 OLS/MLP：0/205，底盘跑飞。
  - target 换到车体坐标系，加 DART：5 cm 25-32；28 维纯变换：33-129，seed 间标准差约 19 个百分点；白盒 32 维：115-162。
  - 1 cm 下最多 23-59；40 s 时没有任何模型在 1 cm 内。
  - 到达后会漂走：中位 4-13 s 后离开 5 cm，保持 ≥ 10 s 的只有 0-21（白盒 61-92）。
  - 随机起点训练（E1）：底盘没问题，但机械臂不开始伸（9 个模型中 8 个），他推测机械臂靠 clock 触发。
  - E3 混合数据：机械臂的触发"按 seed 要么整个学会，要么整个学不会"。
- 按这些先验，我们最可能看到的情况：
  - 底盘能到位，失败集中在"机械臂不开始伸"和"到了又漂走"；
  - 10 mm 成功率可能很低；
  - seed 方差很大。
  这一条写成明确的预测，与下面的假设并列。

## 2. 本步的问题与结构（相对原计划的改动）

- 主线表示用 ego，这是有证据的设计决定，不再作为实验去"试"。证据有三点：
  Rodrigo 的结果；第 2 步发现相对坐标下覆盖密度高 6 倍；任务对平面平移和旋转不变。
  - raw 和 ego 的比较同时改了两件事（坐标不变性和极坐标特征），不能归因，所以 raw 只作描述性基线。
- 预注册实验 A（从第 4 步提前）：ego 对 ego 去掉 (v̂, ŵ)，唯一的变量是是否输入实测轮速。
  - 它检验第 2 步已经写好的 H-copycat、H-hidden-phase、H-goal 三个竞争假设。
  - 结果也回答 Rodrigo 的"底盘静止 / 停稳信号"问题（他的 §6e 候选 (a)，以及 E6 关于触发的问题）。
- 预注册实验 B：底盘/机械臂 2×2 拆分诊断，专家接管一半。Rodrigo 没做过（他的 §6b(ii) 只提了没跑）。
- 描述性部分：
  - raw 基线；线性模型（ego）回答"表示对了之后还需不需要非线性"；
  - 失败分类；seed 一致性；成功率与覆盖度的关系；离线误差与闭环成功率的关系；M1 和 M2 的伸臂触发对比。
- 第 4 步剩下：学习曲线（在 ego 上，横轴是 episode 数）、降采样步长扫描（含抗混叠对照）。

## 3. 前置工作（在看任何模型的闭环结果之前完成）

3.0 协议升到 v1.1（已决定）。主指标完全不变，只加次要指标和修正诊断列：
- 首次到达后继续仿真 10 s（v1 是 1 s）。success 和 settled 仍按前 1 s 判定，与 v1 完全相同；1-10 s 内出错不影响主指标。
- 新的次要指标：
  - held_10s：10 s 内末端误差始终 < 15 mm；
  - 10 s 时的末端误差，hold 期间底盘的位移和最大轮速（看"到了又漂走"）；
  - Rodrigo 的"5 cm 内停留 0.5 s"及其首次发生时刻，用于和他的结果并排。
- 修正诊断列：
  - 原来的 arm_move_time 对专家无效：保持阶段 j2 下垂 0.050-0.058 rad，超过了 0.05 rad 的阈值，800/800 个专家 episode 都在约 0.58 s 记下"开始伸臂"。
    改为 arm_extend_time = 末端在车体坐标系下相对 t = 0 的位移首次 > 10 cm（Rodrigo 的定义）。
    专家导航期间最大 5.4 cm；nav_done 后中位 26.9 cm，p5 4.2 cm，约 5% 的 episode 无法判定。
  - 新增"停靠环距离"：|‖target_xy − base_xy‖ − standoff(z)|，以及朝向 target 的误差。这个指标不依赖专家停靠点的方位：我们的模型拿不到起始位置，停在环上的其他点也够得到。原来以专家停靠点为参照的列保留，作为次要。
- 重跑专家 DEV/HARD/CONFIRM 并加 --compare-recorded：要求主指标逐 episode 与 v1 相同，800/800 逐步一致。约 15 分钟。
- 写进 report.txt 的版本记录。

3.1 训练管线（train_policy.py）与单元测试
- 数据：过滤后的 1897 个 episode，按 episode 划分 85% 训练、15% 验证，按 noisy × mode 分层，划分 seed 固定。
- 降采样：步长 9（约 111 Hz）。每个 episode 的起始偏移用一个独立的数据 seed 生成一次并存盘，所有模型、所有训练 seed 共用同一组训练行。
  - 用奇数步长：实测低 z 时连续抖动步 99% 正负交替；每个 episode 抽到的标签均值相对平滑标签的偏差，步长 10 时中位 0.36 N·m（p90 3.1），步长 9 时 0.05（p90 0.11）。
  - 约 1.77M 训练行。验证集的离线指标在 1 kHz 全量上算（约 280 万行），因为事件那一步占事件后超额误差的 34-51%。
- 输出 9 维：(v, w) + 7 个力矩，按训练集标准化，训练时等权；部署时用 collect_data.wheel_speeds 换回 4 个轮速。
- 特征：
  - ego（22 维）：target 在当前车体坐标系下的 Δx、Δy，target z，平面距离 ρ，cos β、sin β，arm q(7)，arm qd(7)，v̂、ŵ。
  - ego−v̂ŵ（20 维）：去掉 v̂、ŵ。
  - raw（23 维）：base x、y，sin/cos yaw，q(7)，qd(7)，v̂、ŵ，target x、y、z。
  - 全部是观测的纯变换：不用 t，不用专家规则，不用起始位姿。
- 模型：
  - MLP 2×128 ReLU：用 sklearn MLPRegressor 的 partial_fit，每次调用一个 epoch，保留 Adam 状态。不要用 warm_start 加 fit，它每次都会重建优化器。
    random_state 传 np.random.RandomState(seed) 实例；按 episode 划分的验证集做 early stopping，patience 5，自己保存最佳权重。
    实测单线程每个 epoch 约 6.7 s，多个 seed 各用 1 个线程并行。
  - 线性模型：直接用 OLS。在 N ≈ 1.77M 时，alpha ≤ 1e3 的 Ridge 与 OLS 没有区别。
- 部署：模型存成 .npz（权重、scaler、特征说明），policy 用 numpy 前向，约 6 µs/步。
- 单元测试（必须先通过）：
  (a) numpy 前向与 MLPRegressor.predict 的最大差 < 1e-9（最常见的错误是漏了输出反标准化，或输出层误加了 ReLU）；
  (b) (v, w) → 轮速 → (v, w) 往返一致；
  (c) ego 特征在任意平面旋转加平移下不变；
  (d) 符号约定：w > 0 时 yaw 增大，β > 0 时第一个 w 标签为正。

3.2 失败分类与判据（analyze_rollouts.py）：事先定义，先用已知答案的案例验证
- 判定顺序，每个失败 episode 只归一类：
  1. 崩溃：tipped、invalid_action、policy_error。
  2. 从未接近（停靠环距离的最小值 > 0.3 m）：
     - 不出发：路径 < 0.05 m 且转角 < 10°，并排除起点本来就在环上 0.15 m 内的 episode（M0 单独报告）；
     - 跑飞：最终距离 > 起点距离 + 1 m；
     - 其他。
  3. 接近过：
     - 绕圈：累计转角减去所需转角（align_req + orient_req）> 360°；
     - 到了又离开 / 冲过：最终停靠环距离 > 0.3 m；
     - 停到位但机械臂没伸：arm_extend_time 为空，且 |min_error − navdone_ee_err| < 2 cm（每个 episode 用自己的参照值）；
     - 伸了但没到：min_error 在 1-5 cm 之间；
     - 其他。
- 轮速超出专家包络（> 1.2 × 3.0 rad/s）的步数比例只作描述，不参与归类。专家的 wheel_cmd_max 本身就是 3.0。
- 验证（看任何模型结果之前）：
  - 专家的 14 个失败全部判为"绕圈"；
  - zero policy 判为"不出发"；
  - 专家的 786 个成功不命中任何类别；
  - 专家导航期间不算"已伸臂"；
  - arm_extend_time 与专家 nav_done 相差 < 0.2 s。
- 汇总：可以合并成 Rodrigo 的 4 类（arrive & stay / arrive & leave / 底盘到了末端没到 / 底盘没到），方便两条线对照。图上只画合并后的 4-5 类。

3.3 混合 policy（policies/hybrid.py，供实验 B 使用）
- 同时运行 ExpertPolicy 和模型，按格子取两边的轮速和力矩。专家的那一半用 info（动力学模型），report 中说明只是诊断。
- 用 stdout 记录影子 FSM 的 phase 时间线，以及 done 是由阈值还是 25 s 超时触发的；evaluate_policy 会把它写进 logs/。
- 混合 policy 用到时间（25 s 超时），按协议一律用 --no-early-stop。
- 验证：两边都设成专家，在 DEV 上加 --compare-recorded，要求 300/300 逐步一致。

3.4 pilot 冒烟测试（记为"预注册前已看到"）
- 训练 ego 3 个 seed，在 data/pilot 上闭环评估（每次约 4 分钟）。
- 只看量级和 seed 间差异，用来确定 seed 数和最小可检测差异，并检查地板效应。不跑 ego−v̂ŵ，避免实验 A 的效应在注册前被看到。
- 顺带得到 lab demo 用的模型：用 GUI 跑几个 DEV episode。

3.5 冻结预注册：把下面第 4、5 节连同 pilot 得到的数字写进 report.txt，然后才在 DEV 上运行。

## 4. 实验 A（预注册）：ego 与 ego−v̂ŵ，是否输入实测轮速

- 唯一变量：是否输入 v̂、ŵ。数据、划分、训练行、网络、超参数、训练 seed 集合都相同。
- 竞争假设与机制预测（数字阈值在 3.4 之后写死）：
  - H-copycat（轮速有害）：有 v̂、ŵ 时，起步和停车时滞后。
    预测：ego 的"不出发"和"冲过"更多；事件窗口内模型输出相对标签的滞后 k ≈ 1 步，去掉轮速的模型 k ≈ 0；去掉 v̂、ŵ 后成功率更高。
  - H-hidden-phase（轮速有用）：没有 v̂、ŵ 时分不清 align 和 drive（第 2 步：wheel-speed 规则让 drive 中 FSM 记忆造成的不一致从 14.7% 降到 0%）；而且 v̂、ŵ ≈ 0 是"停稳"信号。
    预测：去掉后，边开边大角度转、底盘停靠率更低，集中在 |d0| 小；arm_extend 比例更低（触发更差）；到达后漂移更大。
  - H-goal（主要问题与轮速无关）：两边的转向误差都集中在偏离起点-target 连线的情况（第 2 步：drive 中 41.7% 的不一致来自单帧缺少停靠点方向）。
    预测：停靠环距离差不多，以专家停靠点为参照的距离都偏大，两者差值约为 0。
  - 先验预测（来自 Rodrigo）：两边的底盘都能到位，主要失败是"机械臂不伸"；如果 H-hidden-phase 对，ego 的这一类会明显更少。
- 主指标：DEV success（协议 v1 / v1.1 的主指标）。地板规则（事先写死）：如果两边 seed 平均的成功率都 < 5%，主指标不判定，改用次要指标判定：
  - 末端曾进入 2 cm / 5 cm 的比例（rollouts.csv 的 min_error）；
  - 底盘停靠率（停靠环距离 < 0.1 m 并静止 ≥ 1 s）；
  - arm_extend 比例；
  - held_10s（如果采用了 v1.1）。
- seed：按 pilot 的 seed 间标准差确定，使两边均值之差的最小可检测差异 ≤ 20 个百分点，最少 5 个，最多 8 个。规则现在写死，在看 DEV 之前定。
  参考：按 Rodrigo 的标准差约 19 个百分点，5 个 seed 时最小可检测差异约 34 个百分点。
- 采纳规则（协议第 5 条的具体化）：
  - 默认保留 v̂、ŵ，因为 brief 的 qd_mes 本来就包含轮速。
  - 只有满足下面全部条件才改为去掉：
    (1) 两层 bootstrap（先在每一边内重抽训练 seed，再重抽 episode）的配对差值 95% CI 下界 > 0；
    (2) 净改进中 ≥ 50% 来自假设预测的失败类别；
    (3) 按 d0 分层没有明显变差，任何一层的差值 CI 上界都不低于 −10 个百分点；
    (4) HARD 上方向一致；
    (5) CONFIRM 上用新训练的 seed 复现方向。
  - 如果 ego 明显更好（反方向满足条件 (1)-(4)），作为"保留轮速"的证据记录，也在 CONFIRM 上用新 seed 复现。
- 报告：
  - 每个 seed 的成功率附 Wilson 区间；seed 平均附两层 bootstrap CI；每个 seed 的 McNemar（只作描述）。
  - 跨表示的配对没有对应关系，所以列出全部 seed × seed 的差值。
  - 失败类别按"每个 episode 各类别的 seed 平均比例"汇总。

## 5. 实验 B（预注册）：底盘/机械臂 2×2 拆分

- 在相同的 DEV episode 上跑 4 个格子：
  - 专家底盘 + 专家机械臂：用混合 policy 跑，与录制逐步一致；
  - 专家底盘 + 模型机械臂；
  - 模型底盘 + 专家机械臂：专家机械臂由影子 FSM 的 nav_done 触发；
  - 全模型：实验 A 中采纳的那一边。
- 模型用实验 A 采纳的那一边的全部 seed。两个混合格子都用 --no-early-stop。
- 事先写明的耦合：
  - 专家机械臂开始伸，等于在正确的时刻给模型底盘一个"已到达"信号（Rodrigo：模型靠 arm_q 区分 orient 和 done）；
  - 专家底盘在 done 之后轮速严格为 0，等于给模型机械臂一个完美的静止信号。
  - 所以单换一半的格子会高估各自那一半，交互项有一部分由构造决定。
- 判读表（主结果）：
  - 逐 episode、逐 seed 归因：
    - "模型底盘格"失败记为底盘原因；
    - "模型机械臂格"失败记为机械臂原因；
    - 两个单换格都成功而全模型失败，记为交互。
  - 给出三类的比例，并以独立假设下的期望 p_A·p_B 作对照。
  - 两个单换格各自与专家做配对差值（seed 平均，两层 bootstrap，3 个对比用 Holm 校正），结果落在：只有底盘 / 只有机械臂 / 两者都有 / 都不显著。
- 附加诊断：
  - "模型底盘 + 专家机械臂"格按触发方式拆开报告：阈值触发 / 超时触发 / 未触发。
  - "专家底盘 + 模型机械臂"格记录模型开始伸臂的时刻减去专家 nav_done 的时刻。这就是 Rodrigo 的 §6b(ii)，直接回答"触发失败还是 reach 轨迹失败"。
  - 可选：另一种与专家规则无关的触发方式（底盘速度 < 1 cm/s、角速度 < 0.02 rad/s，持续 0.5 s），只作诊断。

## 6. 描述性部分（不做采纳决定）

- raw 基线：MLP 1 个 seed 加 OLS，只跑 DEV。事先写下停止规则：如果成功率 ≤ 1% 且多数跑飞，不再加 seed。OLS 也在 ego 上跑一次。
- seed 一致性：
  - 用 Cochran Q 检验 seed 之间的成功率是否不同；
  - 归一化 ψ = ψ / ψ_indep，其中 ψ_indep = 1 − Π p_s − Π (1 − p_s)。0 表示结果由 episode 决定，1 表示纯粹是 seed 抽签；
  - 不用"ψ > 某个值"作规则，因为 ψ 由成功率机械决定。
- 成功率 vs 覆盖度百分位（episodes_{dev,hard}.csv 的 cov_pct_world、cov_pct_rel）：对应 brief 的泛化问题。
- 离线验证误差 vs DEV 闭环成功率：所有模型、所有 seed 画在一张散点图上，复现 Rodrigo 的"离线误差预测不了闭环"。
- M1 对 M2 的伸臂触发（预注册的方向预测）：
  - 如果"不伸臂"集中在 M1（nav_done 之前有约 180° 的原地转向，类似 Rodrigo E1 里变长的 orient），支持"长 orient 段"的解释；
  - 如果各模式都一样，支持他的"缺 clock"解释。
  - DEV 中 M1 有 48 个，HARD 中有 66 个。
- 离线：
  - 各输出组的 RMSE，按 phase，事件窗口内和外，与照抄基线对照；
  - 力矩误差按 reach 段 × z 交叉报告，用来检验 Rodrigo 把 done 段误差归因于 q_des 的解释是否成立，还是来自低 z 抖动；
  - 事件窗口内模型输出相对标签的滞后 k（干净 episode、1 kHz、按事件类型）。这只是描述量，"照抄"是否成立只在实验 A 的闭环里判定。
- 每个模型的推理耗时（policy_ms）和参数量，对应 brief 的 efficiency。

## 7. 统计约定

- 单个 seed 的成功率用 Wilson 区间。seed 平均用两层 bootstrap（先抽 seed，再抽 episode），不要把多个 seed 的试验合在一起算 Wilson（伪重复）。
- Holm 校正的比较族：实验 B 的 3 个对比算一族；各分层算一族。实验 A 的主比较单独判定，不做校正。
- 提醒：n = 300 时，单个成功率在 50% 附近约为 ±5.6 个百分点，在 10% 附近约为 ±3.5。结论主要受 seed 方差限制。

## 8. 产出

- 代码：train_policy.py、policies/（ego/raw/linear policy、hybrid）、analyze_rollouts.py，各自带单元测试；evaluate_policy.py 升到 v1.1（如果批准）。
- 结果：
  - results/<model>_<seed>_<set>/：协议格式。
  - results/step3/：summary.txt；图：失败类别叠加在 (d0, θ) 坐标上、2×2 归因图、成功率 vs 覆盖度、离线 vs 闭环。
- report.txt：v1.1 的版本记录；实验 A、B 的预注册（运行前写）；结果；局限（例如 arm_move_time 的问题，混合格子的耦合）。

## 9. 预算与顺序（实测：全部失败时，DEV 约 12.5 分钟，HARD 约 8.5 分钟；训练每个 seed 几分钟）

1. 3.0 + 3.1 + 3.2 + 3.3：写代码和单元测试，跑专家验证。约半天。
2. 3.4 pilot 冒烟测试，同时准备 demo 模型。约 30 分钟。
3. 实验 A：5-8 个 seed × 2 边，在 DEV 上约 2-3.5 小时；HARD 只跑两边各自的 seed，约 1.5 小时；被判 stalled 的 episode 用 --episodes 单独重跑检查，约几分钟。
4. 实验 B：2 个混合格 × 5 个 seed，在 DEV 上，--no-early-stop，约 2 小时。
5. raw、OLS 的描述性运行，约 40 分钟；CONFIRM 复现（新 seed），约 1 小时。
合计约 8-10 小时计算，大部分可以无人值守、过夜跑。

## 10. 已确认的决定（用户，2026-10-29）

1. 结构：ego 作为主线；raw 只作描述；qd_wheel 消融提前为实验 A；实验 B 是 2×2 拆分。
2. 协议现在升到 v1.1：加 10 s 到达后保持的次要指标，修正伸臂诊断列，加停靠环距离；主指标不变；专家基线重跑。
3. 小组协调：已起草给 Rodrigo 的消息（见 RODRIGO_MESSAGE_DRAFT.md，由用户发送）。
   report 用哪个专家定义、哪个主指标，仍待小组决定。第 3 步继续沿我们这条线做，输出保持可对照：
   次要指标里有他的"5 cm 内停留 0.5 s"；失败类别可以合并成他的 4 类。
   暂时不向他要模型文件。
4. seed 数：由 pilot 冒烟测试的 seed 间标准差决定，使最小可检测差异 ≤ 20 个百分点，范围 5-8 个；规则在看 DEV 之前写死。
