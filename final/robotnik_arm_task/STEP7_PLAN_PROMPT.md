# 第 7 步方案：实验 PC（P 在 HARD 与 CONFIRM 上的确认）

本文件是给执行者（Codex）的自包含任务说明，也是实验 PC 的预注册（§3）。用户 2026-11-03 同意做这一步（"继续"）。
工作目录：`/home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task`。
Python：`pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python`。测试用 unittest（`tests/`）。
机器：24 核、30 GB。仿真 `--workers 22`。

## 0. 背景

- 第 6 步实验 P（`ego-novw-trig-pd`：输出 (v, w, q_des - q, qd_des)，固定的专家 PD + 重力补偿换成力矩）在 DEV 上：
  success 70.9% 对 C3 62.1%，Δ +8.8 pp，bootstrap 97.5% [-0.3, +17.1]，Welch 97.5% [-0.6, +18.2]（p 约 0.03），没有满足冻结的
  条件 (1)（Bonferroni 97.5%），判定 KEEP C3，HARD 与 CONFIRM 没有为 P 打开（report.txt `[Results] 第 6 步：实验 P`）。
- 第 6 步的模型都已按冻结的配方训练好：`trained/ego-novw-trig-pd_s0-4.npz`（HARD 用）、`_s100-104.npz`（CONFIRM 用），模型检查
  通过（results/step6/P_model_checks.json）。C3 的 HARD（s0-4）与 CONFIRM（s100-104）运行已存在：`results/ego-novw-trig_s<k>_<set>/`。
- 代码模板：`results/step6/step6_pipeline.py`（`evaluate`、`validate_run`、`stalled_controls` 的做法）、`gp_common.py`
  （`difference`）、`gp_mechanisms.py`、`analyze_rollouts.py`（`paired_diff`、`seed_t_test`、`classify_run`、`metric_matrix`）。

## 1. 硬性规则

1. 不打开 `data/test`。CONFIRM 只在 §3 冻结之后、为 P 打开一次（`--allow-confirm`）。
2. 不修改已有的文件：第 6 步的所有文件（`results/step6/` 下的脚本与结果、`prereg_GP_frozen.*`）、共享代码（`train_policy.py`、
   `bc_features.py`、`policies/`、`analyze_rollouts.py`、`evaluate_policy.py`）、`trained/`、已有的运行文件夹。只能 import 复用。
   新文件全部放在 `results/step7/`（和 `tests/test_step7.py`）。
3. 不 commit、不 push。
4. 冻结之后不改 §3。必须偏离时：记录、按原规则也算、两者都报告。
5. 只按 §3 给出判定，不写解读。
6. 只运行预计 10 分钟以内的命令；正式的仿真链由 Claude 运行（与第 6 步相同的分工）。
7. 只有验证数值不达标才停下；工具或运行方式的错误（路径、参数、从标准输入启动等）修正后重跑，并记录。

## 2. 任务

1. `results/step7/run_expPC.sh` + `results/step7/pc_pipeline.py`：可续跑的链（已完成的运行跳过、中断的文件夹删除重跑、每个阶段开始
   时核对 §3 冻结的 sha256），阶段：
   - `controls`：C3 HARD s0-4 与 CONFIRM s100-104 的 stalled 行，按第 6 步 `stalled_controls` 的同一做法处理（已有的 nostall 重跑
     先查 `results/step3|step5|step6/nostall/`；没有就用 `--no-early-stop` 只重跑 stalled 的 episode，输出到 `results/step7/nostall/`）。
     注意 step6 的 `stalled_controls` 内部有门槛检查（require_gate），不能直接调用；照它的逻辑在 step7 里写一个不带第 6 步门槛的版本。
   - `hard`：P s0-4 在 data/hard 全部 200 个 episode（`--no-early-stop`，输出 `results/ego-novw-trig-pd_s<k>_hard/`，用 step6 的
     `validate_run` 检查）。
   - `confirm`：P s100-104 在 data/confirm 全部 300 个 episode（`--allow-confirm`、`--no-early-stop`，输出
     `results/ego-novw-trig-pd_s<k>_confirm/`）。HARD 与 CONFIRM 之间没有门槛（§3 的主判定合并两个集）。
   - `analyze`：`results/step7/PC_mechanisms.py` 输出 `results/step7/PC_mechanisms.txt/.json`、每个集的完整比较
     `PC_compare_hard.txt`、`PC_compare_confirm.txt`，按 §3 逐条给数字与判定。
2. 合并统计（§3 的主判定需要）：
   - 每个集分别用 `analyze_rollouts.paired_diff`（两层 bootstrap，10000 次，与第 6 步同样的随机数设置）得到 Δ 的 bootstrap 抽样
     d_H[i]、d_C[i]；合并抽样 d[i] = (200 d_H[i] + 300 d_C[i]) / 500（按 episode 数加权，两个集的抽样独立）；点估计同样加权。
     95% CI 取 d 的 2.5 / 97.5 百分位。
   - 合并的 Welch：每个集按 seed 算两组均值差及其标准误 SE_H、SE_C；合并差 = (200 Δ_H + 300 Δ_C) / 500，
     SE = sqrt((200 SE_H)^2 + (300 SE_C)^2) / 500，自由度用 Welch-Satterthwaite（四个方差项），95% t 区间。
   - 单元测试 `tests/test_step7.py`：(a) 用合成数据核对合并公式（已知答案）；(b) 在第 6 步的 DEV 数据上，单集的路径必须逐位复现
     `results/step6/P_mechanisms.json` 的 DEV success 点估计与 95%（level 0.95 重算）区间所用的同一函数输出——即证明 step7 的单集
     计算与 step6 是同一套代码路径（level 设为 0.975 时应与 P_mechanisms.json 中的 lo/hi 完全相同）。
3. 冻结：§3 原文复制进 report.txt，新节标题 `[Methodology] 第 7 步预注册：实验 PC（P 在 HARD 与 CONFIRM 上的确认）（2026-11-03 写下，
   在 P 的任何 HARD / CONFIRM 运行之前；冻结记录 results/step7/prereg_PC_frozen.txt）`，插在 `\n[References] 参考文献（已核对出处）`
   之前（注意：report.txt 正文中别处也出现 "[References]" 字样，必须用这整行定位）；写 `results/step7/prereg_PC_frozen.txt`
   （该节文本、本文件、冻结时间的 sha256）。
4. 冒烟：冻结后，在 `results/step7/smoke/` 下用合成 rollouts 跑通 `analyze`（含判定的各分支），并用 1-2 个 HARD episode 的真实运行
   （输出在 smoke 目录，不得写到正式路径）证明 `hard` 阶段能跑。
5. 写 `results/step7/REPORT_PC.md`（中文）：文件、测试、冻结、冒烟、给 Claude 的正式命令。

## 3. 实验 PC 预注册（冻结时把本节原文复制进 report.txt）

- 问题：实验 P 在 DEV 上的优势（+8.8 pp）能否在从未为 P 打开过的 HARD 与 CONFIRM 上重现？
- 性质：这是看过 P 的 DEV 结果之后设计的新实验，假设来自 DEV；检验只用 P 没有看过的集合。它不改变第 6 步的判定（KEEP C3，
  按冻结规则得出）；报告时 DEV 与本实验一起呈现。
- 比较：HARD：P s0-4 对 C3 s0-4（200 个 episode）；CONFIRM：P s100-104 对 C3 s100-104（300 个 episode）。C3 的 stalled 行换成
  不早停的重跑行（与 D、F、第 6 步相同）。模型都是第 6 步已训练好的，不重训。
- 主指标 success；统计单位 episode；配对（同一集内同一个 episode）。
- 判定（95%，只有这一个比较，不做多重校正）：
  1. 合并（HARD + CONFIRM，按 episode 数加权）的 Δsuccess：两层 bootstrap 与 Welch 的 95% 下界都 > 0。
  2. 一致性：HARD 与 CONFIRM 各自的 Δsuccess 点估计都 > 0。
  3. 风险：合并的 crash + left_or_overshot 增加 ≤ +3 pp（点估计）。
  - (1)-(3) 全部满足 -> "P 在留出集上得到确认"。否则 -> "没有得到确认"，KEEP C3。
  - 镜像：合并的两个 95% 上界都 < 0 -> 报告"P 在留出集上更差"。
- 确认之后的主模型：只有在助教确认"固定 PD 输出层"符合作业要求时，P 才成为主模型；否则 C3 仍是主模型，P 作为留出集确认过的
  另一种动作表示报告。没有得到确认时，主模型是 C3。
- 描述性的预期（来自 DEV，不参与判定）：低处（z < 0.45 m）success 增加、circling 减少、reach_2cm 增加（合并的点估计方向）；
  extended_not_reached 没有确定的变化。每个集与合并的数字都报告（95% CI），并给出 reach_5cm、held_10s、arm_extended_toward、
  crash、policy_ms。
- 功效与局限（事先写明）：合并后 10 对 10 个 seed，与 DEV 相同；HARD 的分布与 DEV 不同（更难），合并假定两个集上的效应可以平均；
  这是对同一个改动的第二次检验（DEV 一次、本实验一次），解读时两者一起看。

## 4. 交付

REPORT_PC.md 写清：新文件、测试输出、冻结的 sha256、冒烟结果、正式命令（例如
`WORKERS=22 bash results/step7/run_expPC.sh all >> results/step7/chainPC.log 2>&1`）。
