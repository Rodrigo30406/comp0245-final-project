[Pre-registration] 第 11 步：三个补充检验 R / S / W（2026-11-06 夜，提交前）

目的：给报告里三句目前证据不够直接的话各补一个单变量检验，不改变主模型（实验线主模型仍是 C3，最终方法仍是
final recipe 的 MLP-Ego-IC）。只用 DEV；HARD、CONFIRM、data/test 都不打开。

一、共同设定
- 数据：实验线训练划分 trained/data_split.json（1613 个训练 episode，stride 9，缓存 results/step3/cache），与 C3 相同。
- 训练：train_policy 的 sklearn MLP 训练器，超参数不变（ReLU，Adam 1e-3，alpha 1e-4，batch 200，float64，
  验证集早停 patience 5 / rel_tol 1e-3，标准化只用原始训练行），种子 0-4。只改下面每个实验声明的那一个变量。
- 评估：DEV 300 个 episode，协议 v1.1，evaluate_policy 默认参数（早停开，--save-traj），与对照 C3 s0-4 的 DEV 运行
  完全相同的设置。对照直接用已有运行，不重跑：
    joint-K0 = ego-novw s0-4（results/ego-novw_s{0..4}_dev）
    joint-K16 = C3 = ego-novw-trig s0-4（results/ego-novw-trig_s{0..4}_dev）
- 统计：episode 为单位；seed 均值；两级 bootstrap（先 seed 再 episode，同一 episode 配对，10 000 次）95% CI，
  以及 seed 层面的 Welch 95% CI（analyze_rollouts.py compare，与第 3 步相同的实现）。
- 指标：success（v1：10 mm 且正常停止）；arm_extended（EE 在基座坐标系中离开起始位置 > 0.1 m，截至 v1 停止点）；
  parked；P(手臂从未伸出 | parked)；失败分类 extended_not_reached（ENR）、circling、parked_no_arm；
  stable_1cm；held_10s；每步策略耗时。
- 训练产生的验证损失会在评估前看到；只记录，不用于任何判断（第 4 步起的惯例：离线指标不选模型）。
  冻结前已看到：R 第 1 个 epoch 的损失（train 0.0386 / val 0.0467）低于 C3 s0（0.109474 / 0.117376；C3 s0 在当前
  环境中逐位复现了这两个数，环境与代码未变）。读法：起始行拷贝与相似的 hold 行冲突、拟合不了，安慰剂行不冲突。不用于判断。

二、实验 R：复制行的特异性（安慰剂对照）
- 被检验的话：IV-D "the fix: one row, re-weighted"。尚未排除的替代解释：任意追加约 1.4% 带关节速度抖动的行
  （一种正则化）就能让手臂启动，与是哪一行无关。
- 唯一变量：K = 16 份拷贝的来源行。C3 用每个训练 episode 的起始行 k0（第一个 nav_done 步）；R 改用一行安慰剂行：
  从 {k : nav_done[k] 为 False 且 k <= k0 - 1000}（底座仍在运动、距锁存至少 1 s）中均匀抽一步，
  随机数 numpy default_rng([11, episode])；若该集合为空但 k0 > 0，则从 {k < k0} 中抽；k0 = 0 的两个 episode
  （747、880，起点即停车位）从 {k >= 1} 中抽并在结果中注明。抖动、K、拷贝行数（25 808）、标签（照录）全部与 C3 相同。
- 模型名 R（trained/step11/R_s{0..4}.npz），5 个种子。实现：train_policy.run_train 的 C3 路径原样调用，只把
  拷贝行的读取函数换成安慰剂缓存的读取（step11.py train-r；train_policy 的读取函数要求 nav_done，安慰剂行不满足）。
  缓存已建：1613 行，规则计数 moving>=1s 1581 / before_k0 30 / k0_zero_any 2，最小距锁存 1000 步。
- 主假设 H-R（特异性）：起作用的是那一行。预测：
    R1 arm_extended 的 seed 均值 <= 10%（C3 约 79%）；
    R2 success 的 seed 均值 <= 5%；
    R3 R - C3 的 success 差：bootstrap 与 Welch 的 95% 上界都 < -40 pp。
- 竞争假设 H-R'（正则化）：R 的 success >= 30% 或 arm_extended >= 50%。
- 读法：R1-R3 全部成立 -> 支持特异性；H-R' 的条件成立 -> "一行"解释被否定，报告必须如实改写；其余 -> 部分支持，
  如实报告数字。另报告 R - joint-K0 的描述性差值。

三、实验 S：在 A-to-B 任务上直接做两网络拆分（老师的建议）
- 被检验的话：摘要和 IV-C 的 "a base/arm network split does not [start the arm]"。目前证据是 B-EM（上界）和障碍任务
  的 H-COPY，不是主任务上的直接检验。
- 网络（复刻第 9 步 L1 / L1-K0 的设计）：
    S-base：输入 ego_novw 的 6 个目标特征（ego_dx, ego_dy, target_z, rho, cos_beta, sin_beta），不含手臂 q/qd、
            不含轮速；输出 v, w；K = 16 份起始行拷贝（该网络无 qd 列，拷贝即 16 份相同行），同第 9 步 L1base。
    S-arm-K16：输入 ego_novw（20 维），输出 tau1..7；K = 16 份拷贝，抖动同 C3。
    S-arm-K0：同上但不加拷贝。
  每个网络的损失只在自己的输出上（标准化 MSE，等权）。策略：轮子来自 S-base，力矩来自手臂网络，(v, w) -> 轮速与
  BCPolicy 相同。
- 两个单元：S-K16 = S-base + S-arm-K16；S-K0 = S-base + S-arm-K0（同一种子共用同一个 S-base，两单元只差手臂网络
  有无拷贝）。种子 0-4。
- 主假设 H-S-COPY（第 9 步 H-COPY 在主任务上的复刻）：拆分本身不能让手臂启动，拷贝可以。预测：
    S1 S-K0：P(手臂从未伸出 | parked) 的 seed 均值 >= 90%，且 success <= 5%；
    S2 S-K16 - S-K0 的 success 差：bootstrap 与 Welch 的 95% 下界都 > +30 pp。
- 次要 H-S-FACT（双侧，不预测方向；障碍任务上的对应预测失败过）：S-K16 - C3 的 success 与 circling 差及 CI。
  只有两种 CI 都不含 0 时才写"拆分改变了成功率/转圈"。
- 描述性 2x2：{joint, split} x {K0, K16} 的 success 与 arm_extended（joint 两格用已有运行；注意 joint-K0 的底座无拷贝，
  split 两格的底座有拷贝，这一不对称在报告中说明）。

四、实验 W：容量
- 被检验的话：结论的 "not for lack of data or capacity" 与 IV-F / 局限里的 "capacity was not tested" 互相矛盾。
- 唯一变量：隐藏层 (256, 256, 256) 代替 (128, 128)（约 13.8 万参数，C3 的约 6.7 倍）。
    W-K0：ego-novw 配方 + 大网络；W-K16：C3 配方（K = 16）+ 大网络。种子 0-4。
- 运行地点：训练和 DEV 评估在 windcharger（UCL CS，同一 pixi lock、同一代码和缓存）。
  冻结前修改（依据框架检查，未看任何新模型结果）：windcharger 上专家 DEV --compare-recorded 为 0/300 逐位一致
  （成功率 298/300 = 99.3%，与本机相同；两机浮点差异被接触动力学放大）。因此 W 不与本机运行比较：
  对照 C3 s0-4 与 ego-novw s0-4 在 windcharger 上用同一评估参数重跑（wc_C3_s*、wc_ego-novw_s*），
  W-K16 vs wc_C3、W-K0 vs wc_ego-novw，比较只在同一台机器内做。另报告：wc_C3 与本机 C3 的描述性差值
  （C3 在第二台机器上的复现），以及 C3 s0 在 windcharger 上两次运行是否逐位一致（机内确定性）。
- 假设 H-W-onset（启动失败不是容量问题）：W-K0 的 arm_extended <= 10% 且 success <= 5%。
- 假设 H-W-floor（最后 1-2 cm 不是容量问题）：W-K16 - C3 的 ENR 份额差点估计 > -5 pp 且 95% CI 含 0；
  success 差的 95% CI 含 0。
- 竞争假设 H-W-cap（容量能推动最后一厘米）：ENR 差 <= -8 pp 且两种 CI 都 < 0，并且 success 差两种 CI 都 > 0。
  成立则报告删去"精度底线不是容量问题"的说法。
- 两者都不成立 -> 不确定，报告如实写"未定"，并相应弱化结论措辞。
- 另报告：每步策略耗时（必须 < 1 ms）与参数量。

五、不做的事与冻结
- 三个实验都不更换主模型，没有采纳规则；三个主假设各自独立，不跨实验做 Holm 校正，次要指标只作描述。
- 冻结：把本文件、新代码和训练命令的 sha256 写入 results/step11/prereg_step11_frozen.txt，并把本节原文插入 report.txt
  的 '\n[References] 参考文献' 之前；之后才对任何新模型运行 DEV。冻结后的任何改动都作为修正案写明时间和原因。
