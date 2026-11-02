# Artifact sync between computers

The repository tracks code, specs and documents. Datasets (`datasets/`, 68 GB) and runs (`runs/`, 12 GB, of which 11 GB
are evaluation traces) are git-ignored and live on the lab PC (`LABIA1395`, 192.168.27.91, SSH on). Its link to the
internet is about 1 MB/s in both directions, so the artifacts move in two tiers (`scripts/sync/`; none of it is part of
the provenance hash, which covers `scripts/*.py` only).

| Tier | What | Size | Where | How often |
|---|---|---|---|---|
| **Light** | training runs (weights, `model_info.json`, metrics, split), `runs/splits`, `runs/diagnostics`, every evaluation without its traces (one `.tar.gz` each), every dataset's metadata (`dataset.json`, `config.json`, `scenarios.json`, `provenance.json`, `episodes/*.json` with the recordings' sha256; one `.tar.gz` each), `docs/figures`, `runs/sync/manifest.json` | ~0.5 GB, ~4 200 files | private Hugging Face dataset repo `Rodrigo30406/comp0245-artifacts` | nightly timer + after each chain |
| **Heavy** | episode recordings (`datasets/*/*/episodes/*.npz`), evaluation traces (`runs/evaluation/*/traces`) | ~79 GB | rsync over SSH from the lab PC on the same network (or a USB disk) | when needed; incremental |

Aborted, interrupted and smoke-check collections are not synced in either tier.

## One-time setup on the lab PC

```bash
pip3 install --target ~/.local/share/hf-tools 'huggingface_hub[cli,hf_xet]'      # done on LABIA1395
PYTHONPATH=~/.local/share/hf-tools ~/.local/share/hf-tools/bin/hf auth login     # paste a WRITE token from https://huggingface.co/settings/tokens
scripts/sync/push_artifacts.sh --list       # what the light tier contains, with sizes
scripts/sync/push_artifacts.sh              # first upload (~10 min at 1 MB/s), creates the private repo
scripts/sync/install_timer.sh               # user systemd timer, every night at 03:30, log in ~/.local/state/comp0245-sync.log
```

The chains (`chain_*.sh`) call `scripts/sync/push_artifacts.sh` at their end once the login exists.

## On the other computer

```bash
git clone https://github.com/Rodrigo30406/comp0245-final-project.git && cd comp0245-final-project && git checkout rodrigo
pip3 install --target ~/.local/share/hf-tools 'huggingface_hub[cli,hf_xet]'
PYTHONPATH=~/.local/share/hf-tools ~/.local/share/hf-tools/bin/hf auth login     # a READ token is enough
scripts/sync/pull_artifacts.sh light        # runs/, dataset metadata, evaluations without traces; unpacks the archives in place
scripts/sync/pull_artifacts.sh heavy        # on the lab network: rsync datasets/ and the traces (resumable; re-run to continue)
scripts/sync/pull_artifacts.sh verify       # sha256 of every recording against its episodes/*.json
```

After `light`, `scripts/analysis/final_table.py`, `summarize_eval.py` and every trained policy work (they read
`runs/`); re-training or re-evaluating needs the recordings from `heavy`. `COMP0245_HEAVY_HOST` and
`COMP0245_HEAVY_PATH` override the lab PC address and path (for a USB copy: `rsync -aP /media/usb/datasets/ datasets/`).

## Files

- `scripts/sync/common.sh` — repo id, host, the light-tier list.
- `scripts/sync/manifest.py` — `runs/sync/manifest.json` (datasets with episode counts and bytes, training runs with
  `model_sha256`, evaluations with trace sizes); `--light` lists the selection; `--verify` hashes the recordings.
- `scripts/sync/pack.py` — the per-evaluation and per-dataset archives (rebuilt only when something inside changed);
  `--unpack` extracts them without overwriting newer local files.
- `scripts/sync/push_artifacts.sh`, `pull_artifacts.sh`, `install_timer.sh`.
