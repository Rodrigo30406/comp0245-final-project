"""Freeze the verbatim approved GP rules only after verification has passed."""
from pathlib import Path
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent
TASK = HERE.parent.parent

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def freeze():
    plan_path = TASK / 'STEP6_PLAN_PROMPT.md'
    plan = plan_path.read_text()
    rules = plan[plan.index('## 6. 预注册'):plan.index('## 7. 分工')].rstrip()
    g5 = plan[plan.index('### G5 '):plan.index('## 4. 实验 P')].rstrip()
    amendment_start = plan.index('- 修订 1（')
    amendment = plan[amendment_start:plan.index('- 损失权重：', amendment_start)].rstrip()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    day = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
    header = ('[Methodology] 第 6 步预注册：实验 G（action chunking）与 P（关节参考输出）\n'
              f'（{day} 写下，在 G、P 的任何 pilot / DEV / HARD / CONFIRM 运行之前；'
              '冻结记录 results/step6/prereg_GP_frozen.txt）')
    section = (header + '\n' + '-' * 120 + '\n' + rules + '\n\n' + g5 + '\n\n'
               + amendment + '\n\n')
    report_path = TASK / 'report.txt'
    original = report_path.read_bytes()
    text = original.decode()
    marker = '[References] 参考文献（已核对出处）'
    if '[Methodology] 第 6 步预注册' in text:
        raise RuntimeError('Step 6 section already exists; refuse refreeze')
    if text.count(marker) != 1:
        raise RuntimeError('Ambiguous report insertion point')
    frozen_path = HERE / 'prereg_GP_frozen.txt'
    hashes_path = HERE / 'prereg_GP_frozen.json'
    if frozen_path.exists() or hashes_path.exists():
        raise RuntimeError('Frozen record exists; never overwrite')
    verification = json.loads((HERE / 'smoke' / 'verification_summary.json').read_text())
    if verification.get('all_required_passed') is not True:
        raise RuntimeError('G4/P2/backward-compatibility not passed')
    new_report = text.replace(marker, section + marker, 1)
    # Existing report bytes are preserved exactly apart from this single insertion.
    assert new_report.replace(section, '', 1).encode() == original
    frozen = ('冻结时间（UTC）: ' + timestamp + '\n'
              '冻结时间 sha256: ' + sha_bytes(timestamp.encode()) + '\n'
              'STEP6_PLAN_PROMPT.md sha256: ' + sha_bytes(plan_path.read_bytes()) + '\n'
              '冻结节 sha256: ' + sha_bytes(section.encode()) + '\n\n' + section)
    hashes = {'created_utc': timestamp, 'time_sha256': sha_bytes(timestamp.encode()),
              'plan_sha256': sha_bytes(plan_path.read_bytes()),
              'section_sha256': sha_bytes(section.encode()), 'report_header': header,
              'frozen_sha256': sha_bytes(frozen.encode()),
              'report_before_sha256': sha_bytes(original), 'section_length': len(section),
              'includes': ['§6 verbatim', 'G5 verbatim', '§4 P1 revision 1 verbatim'],
              'verification_sha256': sha_bytes((HERE / 'smoke' / 'verification_summary.json').read_bytes())}
    report_path.write_text(new_report)
    frozen_path.write_text(frozen)
    hashes_path.write_text(json.dumps(hashes, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(hashes, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    freeze()
