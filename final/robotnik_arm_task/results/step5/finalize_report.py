"""Assemble the Chinese audit report from saved evidence without inferring results."""
import json
from pathlib import Path

from f_common import HERE, TASK, check_freeze, sha


def load(name, default=None):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else default


def main():
    complete = load("chainF_complete.json")
    blocked = load("chainF_blocked.json")
    report = HERE / "CODEX_REPORT_F.md"
    previous = report.read_text()
    history = previous.split("## 首次 F1 会话历史（保留原文）\n\n", 1)[1].split("\n\n## 修订 1 续跑记录", 1)[0]
    plan = load("amendment1_baseline.json")
    freeze = load("prereg_F_hashes.json")
    title = "已完成" if complete else "阻塞，停止执行"
    lines = [f"# 第 5 步执行报告（Codex，修订 1，{title}）", "",
             "时间均为 UTC（本次日期与 Europe/London 相同）。首次 F1 会话全文保留在报告末尾。", "",
             "## 当前状态与计划完整性", "",
             f"- 方案新 SHA256：`{plan['sha256']['STEP5_DAGGER_PLAN_PROMPT.md']}`。",
             f"- 首次方案 SHA256：`{plan['original_plan_sha256']}`。移除新增 F1 修订 1 段后，逐字节恢复该哈希；除此之外方案没有改动。",
             f"- 冻结时间：`{freeze['frozen_at']}`。§3 原文和修订 1 写入 report.txt，并保存为 prereg_F_frozen.txt；冻结节 SHA256：`{freeze['section_sha256']}`。",
             "- 每个阶段开始时检查冻结方案、冻结 report 节、代码、原模型与输入哈希。没有打开 data/test 或 CONFIRM 数据。F0 的已实现路径只会读取已有 CONFIRM 结果，本次尚未运行 F0。",
             "- 没有 commit、push 或更改 git 配置；没有改协议常量、原模型或旧运行文件夹。", "",
             "## 新建与修改的文件", "",
             "新建代码：dagger_relabel.py（上次会话已建）；results/step5/f_common.py、tolerance_curve.py、build_dagger_cache.py、F_mechanisms.py、step5_pipeline.py、run_expF.sh、run_F1_amend1.sh、finalize_report.py；tests/test_extra_rows.py、test_f_gates.py。",
             "修改：train_policy.py、tests/test_dagger_relabel.py（修订 1）；report.txt（冻结节" +
             ("及结果草稿" if complete else "；本次阻塞，没有写结果草稿") +
             "）、README.md（第 5 步运行与输出说明）。方案由用户在本次会话之前加入修订 1，本次没有修改。",
             "train_policy.py 的新增内容：MODELS/TRIGGER_MODELS 注册 dag1；load_extra_rows 校验 manifest、split、特征维度、float64、有限值及数组 SHA256；run_train 的 extra_rows 默认 None，只在显式启用时于触发帧拷贝之后追加、沿用原 stride-9 标准化；meta 记录 manifest SHA256 与行数；train/train-many CLI 增加默认关闭的 --extra-rows 并传给子进程。其余共享代码没有改动。", "",
             "## F1：修订 1 验证", "",
             "首次 F1_verification.json/.log 与 initial_sha256.txt 保留原样。修订后首次测试新增的时序断言错误地要求 nav_done 不晚于 arm_move_time；该测试失败记录 F1_amend1_exact.json 和 F1_amend1_verification.log 也保留。evaluate_policy.py 的 ARM_MOVE_ANGLE 注释已说明：专家的保持姿态下沉会超过 0.05 rad，所以该量对专家不是伸臂触发诊断。修正测试从 float64 观测精确复算 arm_move_time，并验证首次 nav_done 之前影子的参考角保持初始值；没有修改容差或方案规则。", "",
             "最终证据：F1_amend1_exact_v2.json、F1_amend1_recorded_v2.json、F1_amend1_verification_v2.log；2 个 unittest 均通过，13.033 秒。", "",
             "| 验证 | episode | 步数 | 轮速最大误差 rad/s | 力矩最大误差 N·m | nav_done 检查 |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, kind in (("F1_amend1_exact_v2.json", "(a) float64"), ("F1_amend1_recorded_v2.json", "(b) float32")):
        j = load(name)
        for r in j["episodes"]:
            nav = (f"首步 {r['first_nav_done']:.6f}s，arm_move {r['arm_move_time']:.6f}s；复算一致、触发前参考保持"
                   if "first_nav_done" in r else f"逐步不同数 {r['nav_done_mismatched_steps']}")
            lines.append(f"| {kind} | {r['episode']} | {r['n_steps']} | {r['wheel_cmds']['max_abs_error']:.17g} | "
                         f"{r['torque_arm']['max_abs_error']:.17g} | {nav} |")
    lines += ["", "(a) 所有保存观测与标签均为 float64，每步轮速与力矩误差为 0，满足 ≤1e-9。轨迹未保存 nav_done；arm_move 提前与保持下沉一致，逐条从轨迹重算完全等于 rollouts.csv，首次 nav_done 前参考角误差为 0，没有时序矛盾。",
              "(b) 与首次相同的 5 条干净、5 条噪声 train episode；轮速 ≤1e-4、力矩 ≤1e-3，且 nav_done 每步相同，全部通过。", "",
              "## 旧路径不变的证据", ""]
    bc = load("backward_compatibility.json")
    if bc:
        lines += ["全部旧测试连同新增验证见 tests_all.log。C3 seed 0 按原配方重训到 results/step5/C3_retrain_s0.npz，未覆盖原模型。",
                  "```json", json.dumps(bc, indent=2, ensure_ascii=False), "```"]
    else:
        lines.append("尚未完成；不能声称旧路径验证已通过。详见当前阻塞和 tests_all.log。")
        if blocked:
            lines.append("本次实际结果：" + blocked.get("test_summary", ""))
            lines.append("五个错误均发生在 setUpClass：" + "、".join(blocked.get("affected_classes", [])) + "。")
            lines.append("调用链：tests/test_training.py:138 → train_policy.py:517 的原有 build_cache → "
                         "multiprocessing/forkserver.py:167 listener.bind(address)；" + blocked.get("root_cause", ""))
    lines += ["", "## 阶段时间、运行与错误", "",
              "| 阶段 | 状态 | UTC 时间 | 详情 |", "| --- | --- | --- | --- |"]
    for name, kind in (("F1_amend1_exact_v2.json", "F1(a) 精确重放"),
                       ("F1_amend1_recorded_v2.json", "F1(b) 录制重放")):
        j = load(name)
        lines.append(f"| {kind} | 通过 | {j['started']} — {j['finished']} | {len(j['episodes'])} 条轨迹 |")
    lines.append(f"| 冻结 | 完成 | {freeze['frozen_at']} | §3 原文与修订 1 |")
    ep = HERE / "stage_events.jsonl"
    events = [json.loads(s) for s in ep.read_text().splitlines()] if ep.exists() else []
    for e in events:
        detail = {k: v for k, v in e.items() if k not in ("time", "stage", "status")}
        lines.append(f"| {e['stage']} | {e['status']} | {e['time']} | {json.dumps(detail, ensure_ascii=False)} |")
    valid = [e for e in events if e["stage"] == "run" and e["status"] == "validated"]
    unique = {e["folder"]: e for e in valid}
    lines += ["", f"链中已验证的独立运行文件夹数：{len(unique)}；episode 合计：{sum(e['n'] for e in unique.values())}。",
              "F1 专家 pilot 另有 1 个运行文件夹、5 个 episode，全部 reached，harness_error=0、policy_error=0。",
              "初始 nohup 启动未保留执行会话，01:42:07 的无 summary pilot 目录被删除后于 01:42:14 重跑成功；没有覆盖已完成的运行。之后使用 nohup 加 wait 保持执行会话。链中后续被中断与删除的目录详见上表。", "",
              "## F0 一致性", ""]
    f0 = load("tolerance_curve.json")
    if f0:
        lines.append(f0["definition"])
        for s, data in f0["sets"].items():
            for model, r in data["models"].items():
                p = next(p for p in r["curve"] if p["tau_mm"] == 10)
                lines.append(f"- {s} {model}：10 mm {100*p['mean']:.6f}%，检查 {json.dumps(r['checks'])}。")
        lines.append("完整 5/7.5/10/15/20/30/50/100 mm 表格及 2–100 mm 曲线见 tolerance_curve.txt/.json 和 fig_tolerance.png/.pdf；20、50 mm 按单元逐一等于 reach_2cm、reach_5cm。")
    else:
        lines.append("未完成。")
    for title, name in (("F2→F3 pilot", "F_pilot.txt"), ("F3 描述性检查", "F_data.txt")):
        lines += ["", "## " + title, ""]
        p = HERE / name
        lines += [p.read_text() if p.exists() else "未完成。"]
    lines += ["", "## 模型检查", ""]
    mc = load("model_checks.json")
    lines += [json.dumps(mc, indent=2, ensure_ascii=False) if mc else "未完成；15 个 dag1 模型未全部通过检查。"]
    lines += ["", "## §3 的逐条数字与判定", ""]
    fm = HERE / "F_mechanisms.txt"
    lines += [fm.read_text() if fm.exists() else "未获得 F_mechanisms.txt，不能给出采纳或 KEEP C3 判定。"]
    lines += ["", "## 实施修正、偏离与影响", "",
              "- F1 浮点容差变动仅为用户批准的修订 1，冻结前完成。首次规则下的结果仍在历史节，没有用新容差改写旧结论。",
              "- F1 新增时序测试的错误解释已修正，并保留失败记录；数值重放误差和 §3 判定规则没有变化。",
              "- 为先证明共享训练路径不变，执行器把完整测试和 C3 seed 0 重训安排在 F0/pilot 前；实际测试失败，C3 重训尚未开始。F2 pilot→F2→F3→F4→模型检查→DEV→门槛的计划顺序不变，没有 F 模型评估结果提前泄露。"]
    change = load("implementation_change1.json")
    if change:
        lines.append("- 执行器别名校验修正及阶段边界重启：" + json.dumps(change, ensure_ascii=False))
    lines += ["- §3 的规则未作任何修改；实施修正没有产生另一套统计判定规则。", "",
              "## 未完成部分与准确续跑命令", ""]
    if complete:
        lines += ["授权范围内的流水线已按门槛完成：" + json.dumps(complete, ensure_ascii=False),
                  "未通过门槛的后续集合按冻结规则保持关闭；这不属于遗漏。无需继续运行。"]
    else:
        lines += ["当前阻塞：", "```json", json.dumps(blocked, indent=2, ensure_ascii=False), "```",
                  "恢复前须先解决上述错误并核对冻结计划/报告节。不要覆盖首次 F1 文件、完整运行或原模型。未完成且没有 summary.txt 的运行目录由执行器删除重跑。", "",
                  "需要在允许 forkserver 绑定本地 AF_UNIX socket 的执行环境恢复；当前会话的权限策略不允许提权，未尝试绕过沙箱、改用假测试或把失败类跳过。以下命令只有在该环境条件满足后才能推进。先做环境检查与冻结核对：", "",
                  "```bash", "cd " + str(TASK),
                  "MPLCONFIGDIR=/tmp/mplcfg pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python - <<'PY'",
                  "import socket, tempfile", "from pathlib import Path", "import sys",
                  "with tempfile.TemporaryDirectory(prefix='step5_socket_', dir='/tmp') as directory:",
                  "    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:",
                  "        sock.bind(str(Path(directory) / 'forkserver.sock'))", "        sock.listen(1)",
                  "sys.path.insert(0, 'results/step5')", "from f_common import check_freeze", "check_freeze()", "PY", "```", "",
                  "```bash", "cd " + str(TASK),
                  "nohup bash results/step5/run_expF.sh all >> results/step5/chainF.log 2>&1 < /dev/null &",
                  "step5_chain_pid=$!", 'wait "$step5_chain_pid"', "```",
                  "轮询时先 sleep 600，再查看 chainF.log。全部 Python 命令均须带 MPLCONFIGDIR=/tmp/mplcfg 并使用指定 RoboEnv pixi manifest。"]
        lines += ["恢复后不需要重新冻结或改 F1 容差；执行器先重新运行全部 unittest，通过后才重训 C3、运行 F0→pilot→400 rollout→F3→F4→模型检查→门槛评估。当前没有后台进程需要继续轮询，最后一次 600 秒定时检查为 02:02:06 UTC。",
                  "全部链结束后再执行（本次阻塞状态已汇总，不应把它误记为已完成）：", "",
                  "```bash", "MPLCONFIGDIR=/tmp/mplcfg pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python results/step5/finalize_report.py", "```"]
    lines += ["", "## 首次 F1 会话历史（保留原文）", "", history]
    report.write_text("\n".join(lines) + "\n")
    if complete:
        heading = "[Results] 第 5 步：实验 F 结果草稿（Codex，待 Claude 审核）"
        target = TASK / "report.txt"
        text = target.read_text()
        if heading not in text:
            pos = text.index("\n[References]")
            body = heading + "\n" + "-" * 120 + "\n" + fm.read_text() + "\n"
            target.write_text(text[:pos] + "\n" + body + text[pos:])
    # An actual integrity failure is reported above and must not be hidden by
    # overwriting its evidence. Verification errors remain visible to callers.
    check_freeze()
    print(f"Chinese audit report written: {report}")


if __name__ == "__main__":
    main()
