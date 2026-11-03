# 第 6 步执行报告：修订 1 验证通过，已冻结并完成隔离短链

日期：2026-11-02；时间均为 UTC，当日与伦敦时间相同。

**本次完成用户第 4–6 项：首次冻结、G/P 全部阶段冒烟、门槛与续跑验收及交接。正式缓存、G/P 完整训练、G5 / P3 正式 pilot、DEV/HARD/CONFIRM 均尚未运行，交 Claude 执行。** 本轮未修改任何训练、部署或流水线源码，沿用 Claude 18:33 的最终回归和已核对哈希的 C3 / k=1 证据。

## 1. 验证与旧路径不变

Claude 于 18:33 运行并通过的最终命令（依据本次用户明确提供的结果；本轮没有重复运行，也没有编造原始日志或秒级起止时间）：

```bash
pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python -m unittest discover -s tests -f
```

结果：**208 项 OK，2 项原有可选慢测试跳过**。完整回归包括 G4、P2、图表和流水线测试。流水线测试文件实际有 17 个 `test_*` 方法，前次报告的“19项新增测试”更正为 17 项。

| 验证 | 数值与判定 |
| --- | --- |
| P2(a)，修订 1 | 200 状态，reach / 导航各 100；q_hold / q_des 未预先限制，含第 6 关节低于限位；最大差 0 N·m，≤1e-9 |
| P2(b) | 5 clean + 5 noisy train episode，97,887 步；最大差 `1.8619718094470272e-5 N·m`，≤1e-3；原参考 lead >0.15 的 624 步保留 |
| P2(c) | 8,001 步，固定层离线 / 部署最大差 0，<1e-9 |
| G4(a,b) | 1000 个实际训练源行，覆盖 729 episode；983 stride + 17 trigger；Δ=9/27/81、各10段和终端 padding 全部逐位相同，j=0 相同；随机受 padding 影响行分别 8/25/77 |
| G4(c) | 常数模型最大差 `8.881784197001252e-16`；已知序列43步差0；k=1 与 BCPolicy 300步差0；reset、权重、对齐和开头归一化通过 |
| G4(d) | 每个Δ顺序运行8,100步，部署 / 离线向量化最大差 `2.0816681711721685e-17`，<1e-9 |
| C3 seed0 全配方重训 | 所有非meta NPZ数组逐位相同，确定性meta / loss历史相同；11 epoch，best epoch6，val loss `0.06037504777466386` |
| k=1 全量矩阵 | X/Y/Xv/Yv 的shape、float64 dtype、sha256均与实际C3训练输入相同；训练1,790,492行、验证2,802,594行 |

证据：[P2](smoke/P2_revision1_verification.json)、[Claude 18:33 G4](smoke/G4_verification_revision1_20261102T183316383000Z_153650.json)、[C3](smoke/revision1/backward_compat_c3.json)、[k=1](smoke/revision1/backward_compat_k1.json)。C3原兼容性运行：16:54:17–16:57:12，wall175.023 s、训练172.687 s、peak1935.141 MiB；k=1：16:57:30–16:57:32，wall1.818 s、peak2127.590 MiB；这些是原兼容性检查的资源，不能作为正式G/P训练资源。

本次冻结前再次计算三份共享文件sha256，与C3和k=1证据的`shared_code_sha256_after`完全相同：

| 文件 | 当前sha256 |
| --- | --- |
| `train_policy.py` | `441df4894b64cf5dbca595e258bd6068ad7d6f66c715c99681234381cfac6e86` |
| `bc_features.py` | `e96bd3a2c6133dec19fd87c0d1e3107b6a4b4ed7aace5d0e74369cd9208f0dc8` |
| `policies/bc_policy.py` | `7fd1505d593c681a46bca9da418bd656f075d15a87c80af30ba888e18018a38a` |

`gp_training.py`也保持`6ffcccd80dad80c41f485957e67d4546cf01cb40d9b60deb1773dba010181479`。因此没有触发本次重做C3 / k=1 / 全回归的条件。[verification_summary.json](smoke/verification_summary.json) 明确区分本次哈希实测与用户提供的Claude回归结果，不是新运行的回归日志。

## 2. 文件交付与共享改动

本次实际写入：`report.txt`仅插入首次冻结节；新增`prereg_GP_frozen.txt/.json`、`smoke/verification_summary.json`；更新本报告；新增冒烟脚本、日志、缓存、模型与证据，全部位于`smoke/finalization_20261102/`。旧报告完整保存在[before_REPORT_GP.md](smoke/finalization_20261102/before_REPORT_GP.md)，其引用的第一次P2失败报告也继续保留。

以下为第6步累计交付，除上段所列，本次均未改动：

| 文件 | 实现内容 |
| --- | --- |
| `train_policy.py` | 登记两个新模型；新增默认关闭的`--chunk-k`、`--chunk-delta`、`--side-cache`，侧缓存子命令、train-many转发与新表示offline分支；旧模型默认保留原路径 |
| `bc_features.py` | save/load支持显式`meta.output_names`并检查维数；旧meta仍使用原9维名称 |
| `policies/bc_policy.py` | 新增ChunkPolicy环缓冲、固定指数权重、PDPolicy及按表示分流的工厂；旧meta仍创建BCPolicy |
| `policies/pd_layer.py` | 复用专家同一个ArmControllerParams、PD律与Pinocchio重力补偿；按修订1不裁剪参考，仅作最终力矩饱和 |
| `gp_training.py` | 90维chunk / 16维参考标签，源文件/代码/数组哈希manifest，j=0全1kHz验证早停，整块stride-9描述损失，P的1/√2机械臂权重、固定层离线指标 |
| `gp_common.py`、`gp_mechanisms.py`、`G_mechanisms.py`、`P_mechanisms.py` | 冻结校验、C3 stalled替换、配对两层bootstrap / seed Welch、Holm、逐级门槛、双采纳主模型规则 |
| `step6_pipeline.py`、`run_expG.sh`、`run_expP.sh` | 全阶段续跑、内存调度、模型meta核对、门槛自动停止、显式CONFIRM令牌和图表完整性检查 |
| `gp_figures.py`、`gp_smoke.py` | pilot/DEV表/容差曲线/P离线表/G平滑度；隔离且明确标注的软件合成rollout和轨迹 |
| `freeze_gp.py`、`check_backward_compat.py` | 首次冻结及逐位兼容性检查工具 |
| `tests/test_chunk.py`、`test_pd_layer.py`、`test_gp_figures.py`、`test_gp_pipeline.py` | G4/P2、图表、门槛/CONFIRM/内存/冻结/续跑/中断恢复验证 |
| `README.md`、`.gitignore` | 第6步接口、文件、运行和资源说明；排除冒烟、日志、轨迹产物 |

除根目录政策文件和tests外，上表未加目录的G/P辅助文件均在`results/step6/`。`evaluate_policy.py`、`analyze_rollouts.py`、`robotnik_arm_control.py`未改，评估协议常量未改。共享改动的旧路径证据为本节所述默认分支、最终208项回归、C3逐位重训、k=1全矩阵及部署一致性；没有仅凭“默认关闭”宣称兼容。

## 3. 首次冻结与完整性

冻结命令：

```bash
pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python results/step6/freeze_gp.py
```

**冻结时间：2026-11-02T18:35:22+00:00。** 标题按§8，位于第5步结果之后、References之前；逐字复制§6、其引用的G5、§4修订1段落。

| 冻结项 | sha256 |
| --- | --- |
| 方案文件 | `cd4f0143e9af4e9d93c09a266655365e1d267167f944f879a20968e857598387` |
| 冻结节文本 | `e4024931fcf64fd96bfdf048e94b27b47b5c6210dbb36e08147469d6ef9fcc57` |
| 冻结时间原文 | `cf58db92e9a5808b8c84a2107ae873c530b3663fb76b67074591cc2d160d6073` |
| `prereg_GP_frozen.txt` | `880cf8f833a08dd90ecb5e887e623149342cd2f9d9b4c95a4982c98ea5a844fe` |

[冻结TXT](prereg_GP_frozen.txt)、[配套JSON](prereg_GP_frozen.json)、[冻结日志](smoke/finalization_20261102/freeze.log)。独立校验于18:36:35完成，[19项检查全部true](smoke/finalization_20261102/freeze_independent_check.json)：原文、标题、位置和所有哈希匹配；节长3800字符/6879字节；去掉新增节后的report字节与冻结前完全相同，sha256仍为`3eb6c9e425da460e9ed169316f1bf89148e87836ac13e12f5cd15e13ad022af1`。每个冒烟阶段入口均实际打印冻结sha256 PASS；冻结后没有改规则或冻结文件。

冻结前对修订1原清单的855个文件核对：变化0、缺失0。最终复核单列本次授权插入的report，其余受保护清单保持不变；详见`smoke/finalization_20261102/final_integrity.json`。原step6根目录`chainG_status.json`、`chainP_status.json`保留历史停机状态，实际本轮状态在隔离root中，避免覆盖旧结果。完整性范围不扩称为对全部大型未跟踪轨迹的逐文件检查；本轮命令未向那些文件写入。

## 4. 冻结后全部阶段短链

所有本轮缓存、训练、评估和分析输出位于`smoke/finalization_20261102/`。两条pass链共用其`chain_pass/`，用2个train episode、1个验证episode、K16、1 epoch、单线程、WORKERS=2、JOBS=1。阶段脚本来自正式入口；DEV/HARD/CONFIRM的统计输入为标记synthetic的100-episode软件fixtures，每集2个seed；没有读取真实这三个评估集。

本次确切命令（均返回0，分别约22.3 s和15.7 s；540 s仅为命令上限）：

```bash
WORKERS=2 JOBS=1 timeout 540 bash results/step6/run_expG.sh all \
  --smoke-root results/step6/smoke/finalization_20261102/chain_pass --smoke-epochs 1 \
  > results/step6/smoke/finalization_20261102/chainG_pass.log 2>&1
WORKERS=2 JOBS=1 timeout 540 bash results/step6/run_expP.sh all \
  --smoke-root results/step6/smoke/finalization_20261102/chain_pass --smoke-epochs 1 \
  > results/step6/smoke/finalization_20261102/chainP_pass.log 2>&1
```

### 阶段时间与运行数

| 阶段 | G（UTC） | P（UTC） |
| --- | --- | --- |
| cache | 18:35:26.069–18:35:27.389 | 18:35:54.029–18:35:54.539 |
| pilot | 18:35:27.390–18:35:35.863 | 18:35:54.540–18:35:57.023 |
| promote | 18:35:35.867–18:35:36.871 | 18:35:57.024–18:35:57.024 |
| train | 18:35:36.874–18:35:38.878 | 18:35:57.025–18:36:00.031 |
| models | 18:35:38.881–18:35:38.899 | 18:36:00.035–18:36:00.052 |
| offline | 18:35:38.900–18:35:41.283 | 18:36:00.053–18:36:02.072 |
| dev | 18:35:41.285–18:35:42.600 | 18:36:02.077–18:36:03.383 |
| hard | 18:35:42.604–18:35:44.368 | 18:36:03.388–18:36:05.134 |
| confirm | 18:35:44.369–18:35:47.092 | 18:36:05.136–18:36:07.946 |
| figures | 18:35:47.093–18:35:48.407 | 18:36:07.947–18:36:09.117 |

事件完整起止：[G events](smoke/finalization_20261102/chain_pass/results/step6/stageG_events.jsonl)、[P events](smoke/finalization_20261102/chain_pass/results/step6/stageP_events.jsonl)；[pass_summary.json](smoke/finalization_20261102/pass_summary.json)列出每个模型时间/内存/行数、实际pilot和合成统计。

| 内容 | G | P |
| --- | --- | --- |
| 源/trigger子缓存 | 2 train + 1 val，共用 | 校验后复用 |
| 新侧缓存 | Δ9/27/81各1份 | pdref 1份 |
| 实际拟合次数 | 9：6个校准 + 1个逐位复训 + 2个剩余seed | 4：s0、s1、s100、s101 |
| 最终模型meta核对 | 4/4通过 | 4/4通过 |
| 拟合计时合计 / 最大peak RSS | 0.447618 s / 178.262 MiB | 0.154445 s / 173.473 MiB |
| 离线新模型 / C3 | 2份 / 2份 | 2份 / 复用2份C3 |
| 实际pilot | 1个模型×2 episode | 1个模型×2 episode |
| 合成DEV / HARD / CONFIRM | 每集2seed×100行；共600行新模型、600行C3 | 同左；共用C3 fixtures |
| promote | 校准s0/s1复制；s0复训所有非meta数组逐位相同，复制文件sha也相同 | 接口为空操作 |

模型检查覆盖特征集、seed、K、自动σ、split/缓存/源代码哈希、x scaler逐位相同、行数、超参数、输出表示、scope、内存；G另核对k=10、Δ和ensemble m=0.11。所有新G/P模型`subset=true`，不能提升到正式模型路径。小训练资源不能外推为正式资源。

G1小缓存受终端padding影响比例：

| Δ | stride训练源行 | trigger源行 | stride验证行 | 含K16拷贝后的训练行 |
| --- | --- | --- | --- | --- |
| 9 | 18/2253 = 0.7989% | 1/2 = 50% | 9/889 = 1.0124% | 34/2285 = 1.4880% |
| 27 | 54/2253 = 2.3968% | 1/2 = 50% | 27/889 = 3.0371% | 70/2285 = 3.0635% |
| 81 | 162/2253 = 7.1904% | 1/2 = 50% | 81/889 = 9.1114% | 178/2285 = 7.7899% |

这里只描述子集；正式比例由Claude完整构建manifest后报告。

### 实际pilot与软件门槛输入

真实闭环：G Δ9 s0和P s0各2 episode，**均success=0/2、tipped=2/2，harness_error=0、policy_error=0、extended_hold_policy_error=0**。这些仅训练1 epoch的短模型不满足正式P3成功率/侧翻门槛；没有将它们报告成P3通过，也没有据此作G/P效果判定。正式模型与100-episode pilot均待Claude运行。软件门槛fixture与真实闭环数值并列保留在`P_pilot.json`，用于验收控制流；不会覆盖实际rollouts。

G5软件输入（**合成，非正式pilot表**）：C3两个seed均50%，Δ9/27/81各两个seed均90%，按“距最高≤5pp取最小Δ”选9；这不是正式Δ*。P3软件输入为n=100、success90%、crash0、policy_error0，只用于进入后续短链。

### 分析与图表

两条链实际生成各集完整比较和mechanisms、DEV三模型表、G pilot曲线、容差曲线、G平滑度和P离线表；`figures_G_manifest.json`和`figures_P_manifest.json`的要求项目全部complete。PNG已目视核对，标题显著标注SYNTHETIC SOFTWARE SMOKE。P验证子集只有一个clean且z≥0.45的episode，其他分箱为NA，未填造缺失值。

短训练的离线描述（2seed平均，仅作路径验收）：C3/P力矩RMSE为nav_hold 0.363889/6.574521、reach_move 0.386926/6.682458、reach_hold 0.153076/5.820274 N·m；P的e nRMSE为0.942958、0.986316，qd_des nRMSE为1.936172、2.057043。含合成轨迹的平滑度C3/G为0.0210616/0.0105308 N·m，相差−0.0105308；均不能作为正式机制证据。

## 5. 自动停止、CONFIRM与续跑

门槛分支使用独立`gates/<实验>_<case>/`。复用pass的smoke模型/缓存；前置阶段标记均注明`synthetic_precondition=true`并附源marker哈希，随后通过真正的`run_exp*.sh all`运行待测DEV/HARD/CONFIRM与分析。不是14次重新训练或真实评估。两个pilot停止分支复用同模型、同CSV的真实2episode记录，只重定位副本config路径；门槛率明确使用合成值。每case完整命令、起止、退出码、源输入不变校验、停止位置、输出集及CONFIRM令牌位于`gates/*_summary.json`。

### 门槛与拒绝分支的软件冒烟

14 / 14 项预期断言通过。以下全部为合成软件测试；前置阶段复用已完成的1 epoch短链副本，marker明确标记synthetic_precondition。新增真实仿真运行与episode均为0；新增训练为0；所有pass源输入sha256均保持不变。

| 实验 | case | UTC开始 | UTC结束 | 秒 | shell退出码 | 状态 | 创建集合 | token | 后续集隔离 |
|---|---|---|---|---:|---:|---|---|---|---|
| G | confirm_fail | 18:38:02.934 | 18:38:11.869 | 8.935 | 0 | chain/complete after confirm | DEV,HARD,CONFIRM | True | 通过 |
| G | confirm_gate_reject | 18:38:34.723 | 18:38:35.495 | 0.772 | 1 | confirm/blocked | 无 | False | 通过 |
| G | confirm_without_flag | 18:38:26.539 | 18:38:27.507 | 0.968 | 1 | confirm/blocked | 无 | False | 通过 |
| G | dev_fail | 18:37:39.250 | 18:37:42.528 | 3.278 | 0 | chain/complete after dev | DEV | False | 通过 |
| G | dev_risk_fail | 18:38:02.933 | 18:38:07.110 | 4.178 | 0 | chain/complete after dev | DEV | False | 通过 |
| G | hard_fail | 18:38:02.934 | 18:38:09.066 | 6.132 | 0 | chain/complete after hard | DEV,HARD | False | 通过 |
| G | pilot_g_stop | 18:38:26.543 | 18:38:28.411 | 1.868 | 0 | pilot/stopped | 无 | False | 通过 |
| P | confirm_fail | 18:38:18.078 | 18:38:27.423 | 9.345 | 0 | chain/complete after confirm | DEV,HARD,CONFIRM | True | 通过 |
| P | confirm_gate_reject | 18:38:26.548 | 18:38:27.516 | 0.968 | 1 | confirm/blocked | 无 | False | 通过 |
| P | confirm_without_flag | 18:38:34.720 | 18:38:35.487 | 0.767 | 1 | confirm/blocked | 无 | False | 通过 |
| P | dev_fail | 18:38:18.079 | 18:38:22.406 | 4.327 | 0 | chain/complete after dev | DEV | False | 通过 |
| P | dev_risk_fail | 18:38:18.081 | 18:38:22.262 | 4.180 | 0 | chain/complete after dev | DEV | False | 通过 |
| P | hard_fail | 18:38:18.076 | 18:38:24.358 | 6.283 | 0 | chain/complete after hard | DEV,HARD | False | 通过 |
| P | pilot_p_stop | 18:38:26.537 | 18:38:27.503 | 0.966 | 1 | pilot/blocked | 无 | False | 通过 |

#### 条件与统计值（全部为软件fixture，差值与CI单位pp）

DEV主success CI为97.5%；HARD/CONFIRM为95%。拒绝分支直接注入的前置condition用 `*` 标明，不存在数值分析结果。

| 实验/case | 集 | c1 | c2 | c3 | c4 | success差 | bootstrap CI | Welch CI | 风险差 | 镜像 |
|---|---|---|---|---|---|---:|---|---|---:|---|
| G/confirm_fail | DEV | True | True | NA | NA | 60.000 | [49.000, 71.000] | [60.000, 60.000] | 0.000 | False |
| G/confirm_fail | HARD | NA | NA | True | NA | 60.000 | [50.000, 69.000] | [60.000, 60.000] | 0.000 | NA |
| G/confirm_fail | CONFIRM | NA | NA | NA | False | -20.000 | [-28.000, -13.000] | [-20.000, -20.000] | 0.000 | NA |
| G/confirm_gate_reject | DEV | False* | True* | NA | NA | NA | NA | NA | NA | NA |
| G/confirm_gate_reject | HARD | NA | NA | True* | NA | NA | NA | NA | NA | NA |
| G/confirm_without_flag | DEV | True* | True* | NA | NA | NA | NA | NA | NA | NA |
| G/confirm_without_flag | HARD | NA | NA | True* | NA | NA | NA | NA | NA | NA |
| G/dev_fail | DEV | False | True | NA | NA | -20.000 | [-29.012, -12.000] | [-20.000, -20.000] | 0.000 | True |
| G/dev_risk_fail | DEV | True | False | NA | NA | 60.000 | [49.000, 71.000] | [60.000, 60.000] | 10.000 | False |
| G/hard_fail | DEV | True | True | NA | NA | 60.000 | [49.000, 71.000] | [60.000, 60.000] | 0.000 | False |
| G/hard_fail | HARD | NA | NA | False | NA | -20.000 | [-28.000, -13.000] | [-20.000, -20.000] | 0.000 | NA |
| P/confirm_fail | DEV | True | True | NA | NA | 60.000 | [49.000, 71.000] | [60.000, 60.000] | 0.000 | False |
| P/confirm_fail | HARD | NA | NA | True | NA | 60.000 | [50.000, 69.000] | [60.000, 60.000] | 0.000 | NA |
| P/confirm_fail | CONFIRM | NA | NA | NA | False | -20.000 | [-28.000, -13.000] | [-20.000, -20.000] | 0.000 | NA |
| P/confirm_gate_reject | DEV | False* | True* | NA | NA | NA | NA | NA | NA | NA |
| P/confirm_gate_reject | HARD | NA | NA | True* | NA | NA | NA | NA | NA | NA |
| P/confirm_without_flag | DEV | True* | True* | NA | NA | NA | NA | NA | NA | NA |
| P/confirm_without_flag | HARD | NA | NA | True* | NA | NA | NA | NA | NA | NA |
| P/dev_fail | DEV | False | True | NA | NA | -20.000 | [-29.012, -12.000] | [-20.000, -20.000] | 0.000 | True |
| P/dev_risk_fail | DEV | True | False | NA | NA | 60.000 | [49.000, 71.000] | [60.000, 60.000] | 10.000 | False |
| P/hard_fail | DEV | True | True | NA | NA | 60.000 | [49.000, 71.000] | [60.000, 60.000] | 0.000 | False |
| P/hard_fail | HARD | NA | NA | False | NA | -20.000 | [-28.000, -13.000] | [-20.000, -20.000] | 0.000 | NA |

#### Pilot分支的范围

G的pass/stop选择都使用明确标注的合成rates：pass各Δ的两seed平均为0.9，stop各Δ为0.1，C3均为0.5；stop选择Δ=9后停止，不打开DEV。不是正式G5校准。

P真实1 epoch模型的两episode证据为success=0、crash=1.0、policy_error=0、passed=false，原样保存在real_closed_loop。软件pass另用n=100、success=0.9、crash=0；软件stop另用n=100、success=0.05、crash=0来验证原阈值触发拒绝。没有把真实P pilot未通过改写为正式通过。

P pilot停止与CONFIRM拒绝的shell exit=1属于预期分支；driver逐项核对原因、未创建下游集/开启token后才返回通过。

全部源输入副本的源文件sha256在每项前后相同；未修改pass短链、正式模型、正式运行或冻结文件。

详细命令、阶段events、条件数值、模型与缓存复用来源见各case summary及 synthetic_preconditions.json。

G/P四种判定失败分支均得到KEEP C3；DEV失败不生成HARD/CONFIRM目录，HARD失败不生成CONFIRM目录；只有到达CONFIRM的用例才有令牌。P pilot失败与拒绝CONFIRM的非零退出是预期的软件保护分支，已通过对应断言，不是P2/G4等数值验证失败。所有case复用的pass输入sha前后不变。

完整链续跑：18:36:45，两条原pass命令各再次执行1次，返回0，各10个阶段均`skip_complete`，不重复训练、仿真或分析。日志为`chainG_resume.log`、`chainP_resume.log`。阶段级跳过校验源码/settings标记；底层完成运行的重新校验另由下列探针证明，不混为一谈。

运行级探针：[evidence.json](smoke/finalization_20261102/run_resume_probe/evidence.json)，18:37:32.118–18:37:33.721，wall1.604 s。新隔离目录预置无summary的sentinel，实际被删除；真正运行G的2个pilot episode；同参数第二次调用验证后跳过，CSV sha256和mtime完全不变，源smoke模型sha未变。evaluate共调用2次，仅启动1个仿真子进程；三类错误计数均0。该探针2episode也均tipped，不作G5判定。本轮因此累计**3个真实仿真运行、6个episode**，其余门槛rollout均为软件合成或复制已有smoke记录。没有真实运行被外部中断；只模拟并验证了一次无summary目录恢复。

## 6. §6数字、正式状态与偏离

| 条件 | 正式G/P | pass软件fixture（两者相同） |
| --- | --- | --- |
| (1) DEV两个97.5% CI下界>0 | 未运行、未判定 | Δsuccess +60pp；bootstrap [49,71]pp，Welch [60,60]pp；通过 |
| (2) DEV风险增加≤3pp | 未运行、未判定 | crash+left_or_overshot增加0pp；通过 |
| (3) HARD success点差>0 | 未运行、未判定 | +60pp；通过 |
| (4) CONFIRM点差>0且bootstrap95%下界>0 | 未运行、未判定 | +60pp，bootstrap [50,69]pp；通过 |
| 镜像 | 未运行、未判定 | pass为false；dev_fail软件用例验证变差分支 |
| G/P均采纳时的主模型 | 未运行、未判定 | 软件差G−P=0pp，95%CI [0,0]，按规则取G |

上述数字摘自隔离`chain_pass/results/step6/G_mechanisms.txt`、`P_mechanisms.txt`与`GP_main_model_selection.json`；软件字符串ADOPT G / ADOPT P只说明分支可达，**没有正式采纳、KEEP C3或镜像结论**。正式G5表/Δ*、P100episode检查、完整模型训练资源、DEV10seed、HARD/CONFIRM5seed、机制数字、图表和解释均尚未完成。

历史修订与执行问题：第一次P2(b)按原方案参考截断失败，episode4最大误差18.446407330084877 N·m；证据原样保留。Claude在任何G/P训练/闭环之前按用户授权写入修订1，去掉参考截断，修订后满足上述P2门槛。前次18:28的标准输入测试入口使forkserver无法重导入`<stdin>`，属于运行方式错误；Claude18:33用文件模块入口完成全回归，故不再将其视为阻断。用户本次澄清：工具/路径/参数错误修正重跑，数值验证不达门槛才停。本轮有一次只读查看把case摘要路径写错（实际为`gates/G_dev_fail_summary.json`），更正后读取，无实验影响；没有用改断言、降门槛或改正式规则绕过验证。

冻结后的协议规则没有偏离。按§7授权用2train/1val、1epoch、2pilot与合成门槛替代正式规模，仅用于软件验收，影响范围以上已列；不产生科学结论。没有打开`data/test`，没有为G/P打开真实DEV/HARD/CONFIRM，没有覆盖已有正式结果/模型/冻结文件，没有commit/push或修改git配置。所有Python均使用指定pixi manifest与真实文件入口；全部本轮实际命令远小于10分钟，分支/仿真命令上限540s。

## 7. Claude正式续跑命令

审核本报告与改动后，由Claude顺序执行；执行者本轮没有运行这些正式长命令。正式all会每阶段核对冻结，按原规则选择Δ、检查P3、训练、补C3必要对照、评估并在门槛失败时停止；CONFIRM只能在前序门槛通过后由脚本显式开启。

建议**WORKERS=22、JOBS=1，两条链顺序运行**：24核留2核，正式G/P完整模型尚无peak实测，1epoch子集的约173–178MiB不能用于提高正式并发。首个完整模型记录peak后，脚本按`(MemAvailable−2GiB)/peak`与请求JOBS取小、最多5进程；任一训练进程>6GiB停止报告。暂时保留JOBS=1最稳妥。若Claude决定并行两条链，仿真各用WORKERS=11，并核算两链合计内存；不要训练和仿真同时满载。G完整离线实现还会分配约1.88GiB的90维预测数组，应计入资源预算。

正式续跑直接重复下面命令即可跳过已完成阶段；不要使用旧smoke异常状态中缺少`--smoke-root`的通用resume字符串来恢复冒烟。冒烟必须带本报告第4节的完整隔离参数；正式命令不带这些参数。

```bash
cd /home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task
WORKERS=22 JOBS=1 bash results/step6/run_expG.sh all >> results/step6/chainG.log 2>&1
WORKERS=22 JOBS=1 bash results/step6/run_expP.sh all >> results/step6/chainP.log 2>&1
```
