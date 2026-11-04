[Methodology] 第 9 步预注册：障碍任务 B1（岔路口）+ 两网络分解 E1（2026-11-04 写下，在任何学习器在 dev2 / confirm2 / test2 上运行之前；冻结记录 results/step9/prereg_step9_frozen.txt）
------------------------------------------------------------------------------------------------------------------------
0) 动机：讲师认为 A-to-B 任务简单，允许定义更难的问题、修改专家（加传感器、加障碍物）；教授建议用两个 MLP（底盘 / 机械臂）
   处理机械臂启动的 copycat。B1 在起点与停靠点之间放一个静止圆柱；在"模糊带"里专家随机从左或右绕行。MSE 回归给出条件均值，
   左右两种示范的均值指向障碍物本身，所以预期无记忆的 MSE 策略在模糊带里撞上或停在障碍物前；"锁定一侧"的底盘头应能消除它。
   同一任务也检验教授的拆分（底盘网络看激光雷达，机械臂网络不看）。方案 STEP9_PLAN_PROMPT.md；实现记录 results/step9/REPORT_A.md、
   REPORT_B.md；所有新代码在 obstacle_task/，不改任何已有文件（快照 /home/w/comp0245/codex_audit/pre_step9_sha256.txt 审核：0 个已有文件变化）。
1) 任务（obstacle_task/scenario.py，冻结）：起点在半径 2 m 圆盘内、朝向任意；target 距起点 3.5-5.0 m、方位任意、z 0.2-1.2 m；停靠点与朝向
   与 A-to-B 专家完全相同（expert_base_goal）；一个圆柱（半径 0.25-0.35 m、高 1.5 m）位于起点-停靠点线段 0.3-0.7 处，横向偏移
   d ∈ [-0.4, 0.4] m（直线必撞）；|d| < 0.2 m 为模糊带（约 48%），带内绕行侧为公平硬币，带外取较短侧 -sign(d)；途经点离障碍表面 0.9 m；
   拒绝采样保证起点、途经点、停靠点的原地转向与两段直线都不碰。每个 episode 独立随机流 [seed, episode, 9]，不影响任何 A-to-B 集合。
   传感器：64 束解析平面激光雷达（底盘中心，基座坐标系，0 = 正前方逆时针，4 m 截断），与 pybullet rayTestBatch 一致到 0.1 mm。
2) 专家 v2（obstacle_task/expert.py）：底盘两段，各段都用未修改的 base_controller（段 0 目标 = 途经点，到达即换段、不执行 orient；
   段 1 = A-to-B 导航）；导航超时 45 s；机械臂 arm_controller 不变，段 1 done 后伸臂。验证（REPORT_A.md）：pilot2 100 个干净 episode
   success 100/100、碰撞 0、实际绕行侧 = 场景侧 100/100、docked_clean 100/100、最小间隙 0.221 m；全噪声 pilot2n 100/100 到达、碰撞 0；
   无障碍模式与 collect_data.py 逐位相同（3 个带噪 episode）；评估程序的专家与录制逐位相同（100/100）。
3) 协议 v2.0（obstacle_task/evaluate.py）：集合 train2（seed 3，2000，50% 带噪）、pilot2（3000，100）、dev2（300，300）、confirm2（302，300）、
   test2（301，300，封存，只为最终冻结模型构建并打开一次）。success = 60 s 内某时刻末端误差 < 10 mm、此前无碰撞、正常结束；碰撞（机器人与障碍物
   接触流形非空，接触间距约 3.5 mm 以内即算）立即结束 episode 并判失败；1 s / 10 s 保持窗口同 v1.1（窗口内碰撞只判保持失败）；
   稳定到达的时限 = T_MAX 60 s（整合时决定，在任何学习器运行之前）。新列：collision、collision_time/link、min_clearance、passed_obstacle、
   side_taken、docked（= v1.1 parked）、docked_clean、time_to_dock、ambiguous、scenario_side、lateral_offset、obstacle_r、policy_latched_side、
   policy_p_left。失败类（classify.py，先匹配先得）：crash、collision、no_pass（从未越过障碍物且未碰撞），其后为 A-to-B 各类。
4) 学习器（obstacle_task/train.py、policies.py）：train2 中 status = reached、未碰撞、导航未超时的 episode（与 A-to-B 的过滤一致；
   导航超时的 episode 的"触发帧"是计时器强制的停车，不是到位），按 episode 85/15 划分（按 noisy × ambiguous 分层，seed 0），stride 9 +
   每个 episode 的偏移；验证 = 验证 episode 的全部 1 kHz 行；激光雷达由记录的底盘位姿离线重算（与部署逐位相同）。sklearn 2x128 ReLU，
   Adam 1e-3、alpha 1e-4、batch 200，各网络按自己的验证损失早停（patience 8、相对 1e-3、最多 150 epoch，保留最佳）。触发拷贝 K = 16
   （与 C3 相同的 qd 抖动规则，占训练行约 1.3-1.5%，与 C3 和 Rodrigo 的最优比例一致）。
   - L0 单网络：single84（arm20 + 雷达 64）-> (v, w, tau1-7)，有拷贝。
   - L1 两网络：底盘 base70（目标 6 维 + 雷达 64，无机械臂状态）-> (v, w)；机械臂 arm20 -> tau1-7；同样的拷贝行。
   - L1-K0：L1 的底盘网络 + 不带拷贝的机械臂网络。
   - L2 锁定侧：L1 的机械臂网络；底盘 = 侧分类器（MLPClassifier 2x64，t < 1 s 的行，标签 = 场景侧）+ 以侧为条件的底盘回归器
     （base70 + 侧）；部署时在第一步算 p_left，>= 0.5 取左，整局锁定。
   dev2 用 seed 0-4（每种配置 5 个），confirm2 用新 seed 100-104。
5) 统计：episode 为单位，所有配置在相同 dev2 episode 上配对；每种配置 5 个 seed；两层 bootstrap（先 seed 后 episode，10 000 次）与按 seed
   成功率的 Welch t，95% CI；每个假设内的次要量 Holm 校正，描述性。模糊带 = ambiguous 列（|d| < 0.2 m，由采样器固定）。
6) 假设与判定
   H-MM（多模态，主假设）：L1（MSE 底盘）在模糊带内把左右平均，撞上或停在障碍物前；L2（锁定侧）不会。
   - 主指标：模糊带内 docked_clean 比率，L2 - L1：bootstrap 与 Welch 的 95% 下界都 > 0 -> 支持。
   - M1：带内 L1 的失败中 >= 60% 是 collision 或 no_pass。
   - M2（特异性）：带外 docked_clean 的 L2 - L1 两个 CI 都在 ±7 pp 之内。
   - M3：L1 内部，带内 docked_clean 低于带外（按 seed 配对，点估计）。
   - M4：L2 越过障碍物时走锁定侧的比例 >= 95%（不来回切换）。
   - 竞争假设：C1 闭环对称性破缺挽救了 MSE（L1 带内 ≈ 带外）；C2 感知是瓶颈（L1、L2 带内带外同样失败）；C3 侧分类器或条件回归器在分布外失败
     （L2 带外更差）。
   - 确认（仅当主指标成立）：confirm2 上 L1 对 L2，seed 100-104：点估计 > 0 且 bootstrap 95% 下界 > 0 -> "得到确认"。
   H-FACT（教授的分解）：L1 对 L0（都有拷贝），双侧，报告 CI；预测整体 docked_clean L1 >= L0（底盘网络用不了机械臂状态的捷径），parked_no_arm
   无差别；次要：success@10 mm、collision、circling。
   H-COPY（教授的 copycat 说法）：L1-K0 对 L1。读数 = 底盘停好（docked）的单元中失败类为 parked_no_arm（机械臂从不启动）的比例（按 seed
   计算后取平均；底盘没停好时"不启动"无从发生，所以用条件比例；无条件的 parked_no_arm 比率同时报告）。预测：L1-K0 >= 50%、L1 <= 10%，
   即单靠拆分不能让机械臂启动；若 L1-K0 < 20%，则拆分本身解决了它（教授是对的），如实报告；介于两者之间记为"两种预注册结果都不是"。
7) 每种配置与专家都报告：success@10/20/50 mm、docked_clean、collision、no_pass、失败类、held_10s、policy_ms；图：docked_clean 与 collision 对横向
   偏移 d（分箱，每种配置带 seed CI，"分岔图"）、模糊带 episode 的俯视轨迹、失败类条形图。test2：最终冻结模型（主指标确认则 L2，否则取 dev2 上
   L0 / L1 中较好者）与专家各跑一次。
8) 事先写明的局限：一个障碍物、静止、已知形状；激光雷达理想化（无噪声、不见自身）；专家用特权信息规划；新任务的数字与 A-to-B 不可比；
   机械臂最后 1-2 cm 的精度问题（A-to-B 的 ENR）仍在，所以 10 mm 成功率不是 H-MM 的主指标；5 个 seed 时约 10-13 pp 以下的差别分辨不出。
9) 冻结前已看到的（如实记录）：train2 / dev2 / confirm2 的专家录制与专家在 dev2、confirm2 上的评估（各 300/300 成功、0 碰撞、与录制逐位相同，
   失败判据在专家成功上都不触发）；30 个模型的训练日志与离线验证损失（L0 0.034-0.038；L1base 0.108-0.113；L2base 0.053-0.057，约为 L1base
   的一半，与"侧别解释了底盘标签的大部分方差"一致，但离线损失不是闭环结果；L1arm 0.023-0.027；L1armK0 0.012-0.013；L2cls log-loss 0.428-0.437）。
   冻结时没有任何学习器的闭环运行（dev2 / confirm2 / test2 上都没有）。整合时的决定（在看到上述任何训练结果之前）：稳定到达时限 60 s、接触判定为
   接触流形非空、训练集排除导航超时（实际 0 个）、H-COPY 读数改为"停好的单元中机械臂从不启动的比例"。
