# 第 7 步实验 PC 执行报告

本轮范围为 `STEP7_PLAN_PROMPT.md` §2 第 1–5 项。已实现链脚本与统计分析，完成 23 项测试、冻结、六组合成分析和真实 HARD 冒烟。正式 controls / HARD / CONFIRM 仿真交给 Claude。

## 文件

- `pc_pipeline.py`、`run_expPC.sh`：`controls → hard → confirm → analyze`。每个阶段开始、每次评估前核验冻结；完整运行经第 6 步 `validate_run` 验证后跳过；中断目录只在本链已登记所有权、模型哈希和 episode 请求相同的情况下删除重跑。既有未登记目录保留。
- `PC_mechanisms.py`：各集完整比较及合并统计，正式运行后生成 `PC_mechanisms.txt/.json` 和 `PC_compare_hard.txt/.json`、`PC_compare_confirm.txt/.json`。单集直接调用第 6 步 `gp_common.difference`；§3 的三项条件与镜像规则分别输出。
- `pc_freeze.py`：冻结、阶段哈希核验、已有文件基线及完整性核验。
- `pc_smoke.py`、`smoke/`：六组合成完整分析和真实 HARD 的两 episode 冒烟。
- `../../tests/test_step7.py`：合并已知答案、DEV 逐位复现、判定边界、续跑与冻结测试。
- `existing_files_baseline.json`、`prereg_PC_frozen.txt/.json`、`prereg_PC_time.txt`、`report_PC_section.txt`、`report_PC_frozen.txt`：完整性基线与冻结产物。
- `implementation_PC_amend1.json/.sha256`、三份 `*.before_amend1.py.txt`：工具兼容修正的前后哈希、原实现备份和原因。

`controls` 对 C3 HARD s0–4、CONFIRM s100–104 的 stalled 行，依次检查 step3、step5、step6 的已有 nostall 重跑，再检查 step7；缺失时只重跑 stalled episode，带 `--no-early-stop`，输出 step7。没有调用第 6 步 `require_gate`。P 的 HARD 固定 200 episode × 5 seed、CONFIRM 固定 300 episode × 5 seed，二者之间没有结果门槛；CONFIRM 需要显式 `--allow-confirm` 并保留一次性开启记录。

## 统计与测试

主统计按 200:300 合并点估计和独立 bootstrap 抽样，10000 次、基准 seed 0，通过 `SeedSequence(0).spawn(2)` 生成两集独立流。Welch 合并四个 seed 方差项，使用各自的 `n−1` 计算 Welch–Satterthwaite 自由度。主判定使用 95%，不做多重校正；风险阈值严格为增加 ≤ 0.03。低 z（<0.45 m）的描述性合并使用两集各自低 z episode 数作权重，单列标明。

真实文件测试入口：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=results/step7/mplcfg \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml \
  python -m unittest tests.test_step7 -f
```

冻结前结果：`Ran 20 tests in 0.357s`，`OK`，日志 `test_step7_final.log`。核心结果包括：

- 合成已知答案：合并 Δ=1.6，bootstrap 95% [1.62, 2.38]；Welch 方差项 [0.16, 0.16, 1.44, 1.44]，SE=√3.2，自由度=100/41。
- DEV 单集：level=0.975 的点估计、区间及全部原数字字段与 `results/step6/P_mechanisms.json` 完全一致；level=0.95 与同一个第 6 步函数、同一 bootstrap 抽样逐位一致。
- 正下界、各集正点估计、风险 +3 pp 包含边界、镜像负上界、已完成跳过、登记中断重跑、外来目录保护、冻结改动拒绝、CONFIRM 一次开启均通过。

实现修正后的追加测试结果和日志在最终核验段列出。

## 冻结与已有文件保护

冻结时间 UTC：`2026-11-03T13:38:19.228412+00:00`；真实 P HARD 冒烟始于 `13:38:39+00:00`。

| 对象 | SHA256 |
| --- | --- |
| 任务说明 `STEP7_PLAN_PROMPT.md` | `38de0c766e44e178748352e1b693b232eab4f056907b2bb1ac739aeddbfe1fea` |
| §3 原文字节 | `2a0ddd7077711e718059ff16c67601882e6015bf0de1bb4efb2f81202654b814` |
| 含指定标题的报告节 | `db98e8e27261de3f3587625eaaa84f16e6d45d97ab55b24340cbf4fb4525990e` |
| 冻结时间文件（含尾换行） | `775015a665e0eaf84020895d59dd82fe9bc1ebc6a22af4aef5526f8849527655` |
| `prereg_PC_frozen.txt` | `ea5730217df041cfdebf8f5fa18e37acd88f214611da0ab79dca02fbe9a3f7fa` |
| 原 `report.txt` | `f0e19e9754118857f47018c8c072f8fcaf728d41d9bb53ec463f391792e0c634` |
| 完整报告副本 `report_PC_frozen.txt` | `af910f7b30b8663fe87ea71a446e5d87009cfe711f8b7402b7ade659d5d4dbaa` |
| 已有文件基线 | `b41d9de562d6eea49e6f7ef0c62a3030af2f8073663b11067ef71deafb0f5ab7` |

**文件规则冲突的处理：** §2.3 要求改 `report.txt`，本轮用户明确禁止修改任何已有文件，因此原文件保持不变。指定新标题和完整 §3 原文写入 `report_PC_section.txt`，另在 `report_PC_frozen.txt` 完整副本中，准确定位整行 `\n[References] 参考文献（已核对出处）` 插入；正文其他位置的 `[References]` 字样没有用于定位。该偏离已写入冻结记录。§3 内容、阈值、合并权重和判定全部按原文执行。

基线覆盖 64,230 个文件／链接，26,111,065,957 字节全量 SHA256，没有未哈希文件。范围包括根文件、已有 results、policies、trained、tests 等；不遍历数据目录，不跟随数据符号链接。初始与冻结前完整核验通过。每个阶段另外核验冻结的全部 PC 源码、共享依赖和 20 个已有 C3/P 模型。

## 冒烟与工具错误修正

真实冒烟命令：

```bash
WORKERS=2 timeout 570s bash results/step7/run_expPC.sh \
  hard --smoke-hard --episodes 0,1
```

输出 `smoke/real/ego-novw-trig-pd_s0_hard/`，与正式 HARD 共用 `evaluations → evaluate → validate_run` 路径。2026-11-03 13:38:39–13:38:48 UTC 完成：覆盖 episode [0,1]；success 2/2、held_10s 2/2；harness_error、policy_error、extended_hold_policy_error 均为 0。实际运行命令带 `--no-early-stop`，输出、日志、状态、所有权登记均位于 smoke 目录。该两 episode 结果仅作程序冒烟。

保留的工具错误及修正：

1. 初始文件查找时 step7 目录、`tests/__init__.py` 或某些 nostall 目录不存在，后续使用实际路径和模块入口继续。
2. 初始导入 matplotlib 的默认 HOME 缓存不可写，自动转入临时目录；后续显式设置 `MPLCONFIGDIR`。没有改写已有项目文件。
3. 第一轮扩展单测的合成冻结夹具未提供新增的冻结输入文件，报 `FileNotFoundError`；补齐夹具后，20 项通过。错误日志保留为 `test_step7_full.log`。
4. 冻结后六组合成分析在完整描述性 `ar.compare_arms` 中报 `TypeError`：其内部 `strata_analysis` 对种子做整数加法，不能接收 `SeedSequence`。原失败日志保留为 `smoke_<case>.log`。amend1 将完整描述性比较的种子转成该流确定生成的 uint32 整数，并在输出标明；PC 的主 bootstrap、合并、Welch、阈值及判定函数保持原计算路径。原冻结 JSON/TXT、§3 和冻结时间保留，源码修改由独立 amendment 与旧文件备份校验。

合成重跑使用完整 HARD 200、CONFIRM 300 个合成 episode，各 5 个 seed；每个 case 生成自己的 `PC_mechanisms` 与两集完整比较文件，均标注为软件夹具。各分支的最终数值见下方最终核验。

## 给 Claude 的正式命令

本轮没有启动正式 controls / HARD / CONFIRM 仿真，也没有开启真实 P CONFIRM。正式运行从工作目录执行：

```bash
WORKERS=22 bash results/step7/run_expPC.sh all --allow-confirm \
  >> results/step7/chainPC.log 2>&1
```

按阶段执行或续跑：

```bash
WORKERS=22 bash results/step7/run_expPC.sh controls --allow-confirm
WORKERS=22 bash results/step7/run_expPC.sh hard
WORKERS=22 bash results/step7/run_expPC.sh confirm --allow-confirm
bash results/step7/run_expPC.sh analyze --allow-confirm
```

首次 CONFIRM 在第 7 步冻结核验后开启，后续同一记录续跑，已完成运行只校验并跳过。正式 P 输出按方案写入 `results/ego-novw-trig-pd_s<seed>_hard/` 和 `_confirm/`，新控制组重跑在 `results/step7/nostall/`。最终仅按 §3 输出“P 在留出集上得到确认”或“没有得到确认，KEEP C3”，另列镜像条件；第 6 步的 KEEP C3 保持原判定。没有得到正式结果前，不给实验 PC 的最终判定。

## 最终核验

追加实现修正后的真实入口结果：`Ran 23 tests in 1.476s`，`OK`，退出码 0；日志 `test_step7_amend1.log`。新增测试实际运行完整 `_compare_set`，验证整份比较文件生成；与原函数逐位核对主统计及 draws；修正前后 `matrix_difference`、`single_set`、`combine_statistics`、`conditions` 四个函数的 AST 完全相同；修订的原冻结锚点、源码哈希与旧版备份核验均通过。

实现修订记录 SHA256：`9a40f3ab084730ae7fcd09648201d579e8df71c0f9559d2c505e9b3a029ddae1`，见 `implementation_PC_amend1.json/.sha256`。原冻结记录仍为上表中的 `ea5730…`，§3 与原冻结时间没有改写；阶段核验同时检查原冻结和修订后的源码。

六组合成分析全部退出码 0，`verification.json` 均 `passed=true`，每组各有完整的两集比较与合并分析。以下数字均为**合成软件验证**，单位 pp，不能作为实验 P 的留出集结论。

| 合成分支 | HARD Δ | CONFIRM Δ | 合并 Δsuccess | bootstrap 95% | Welch 95% | 合并 Δrisk | 条件(1)/(2)/(3) | 夹具判定 |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| confirmed | +60 | +60 | +60 | [55.6, 64.2] | [60, 60] | 0 | 真/真/真 | 得到确认 |
| bootstrap_not_positive | +0.5 | +0.333333 | +0.4 | [0, 1.0] | [0.4, 0.4] | 0 | 假/真/真 | 没有得到确认 |
| hard_nonpositive | 0 | +60 | +36 | [32.6, 39.4] | [36, 36] | 0 | 真/假/真 | 没有得到确认 |
| confirm_nonpositive | +60 | 0 | +24 | [21.2, 26.6] | [24, 24] | 0 | 真/假/真 | 没有得到确认 |
| risk_fail | +60 | +60 | +60 | [55.6, 64.2] | [60, 60] | +10 | 真/真/假 | 没有得到确认 |
| mirror_worse | −70 | −70 | −70 | [−74, −65.8] | [−70, −70] | 0 | 假/假/真 | 没有得到确认；镜像“更差” |

重跑日志为 `smoke_<case>_amend1.log`，输出位于 `smoke/<case>/results/step7/`，原失败日志单独保留。可复现单组：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=results/step7/mplcfg \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
timeout 570s pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml \
  python results/step7/pc_smoke.py --case confirmed
```

真实冒烟重复执行同一命令，日志 `smoke_real_hard_resume.log` 显示 `skip_complete`，覆盖 2 个 episode，三类运行错误均为 0，没有重跑或改写完整运行文件。

最终完整性检查通过：64,230 个已有文件／链接，26,111,065,957 字节与基线相符，无遗漏哈希、修改、删除或范围外新增文件；见 `integrity_final.log`。未打开 `data/test` 或真实 `data/confirm`，未启动任何正式 controls / HARD / CONFIRM 仿真，未创建真实 P CONFIRM 开启记录，未 commit、未 push。所有命令预计在 10 分钟内；较长的冻结、完整性及冒烟调用设有 570 秒超时。机器可读汇总见 `smoke/verification_summary.json`。
