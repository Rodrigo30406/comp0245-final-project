# Working across two computers

## Current storage and publication state

Code, recipes, tests and maintained documentation belong in Git. Generated
`runs/`, `datasets/`, `data/`, `checkpoints/`, `.pt`/`.pth` weights, local `.env`
and `.venv` are ignored. Results exist on disk, but Git push/pull does not move
them. No artifacts from those ignored directories are currently tracked.

The current implementation also contains uncommitted and untracked source files.
A clone of the remote repository alone does not contain the complete current
pipeline. Review and publish the code separately before using that clone as the
second machine's source. No commit or push is performed by this guide.

## Intended division of work

Use the powerful machine for headless collection and large autonomous evaluation
banks. Use the laptop for implementation, analysis and selected training runs.
PyTorch training may also run on the other GPU if desired; profile each host
rather than carrying over the laptop's 16-worker setting as an optimum.

Each machine has its own checkout and locally installed RoboEnv/assets/environment.
Exchange code through Git and exchange complete datasets/model/evaluation run
folders separately. Keep immutable artifact IDs and checksums. Do not let both
hosts write into the same experiment directory or modify source during a job.
Run long jobs from a frozen checkout/worktree of a known commit, while development
continues elsewhere. Record commit, dirty source snapshot where applicable,
configuration, seeds, versions and hardware with each run.

Whole-folder copies matter: a dataset includes configuration, schema, scenarios,
provenance, episode metadata and NPZ arrays. A trained model used by the current
validator needs its run/split/config/model metadata and source dataset, not only
`model.npz`. Evaluation runs include jobs, protocol, summaries, records and traces.
Partial live copies are not complete datasets or scientifically independent runs.

## Portability work still required

The experiment manifests currently store absolute source-dataset and training-run
paths. The frozen learning-curve orchestrators also reference local historical
runs. Merely copying folders to a different home directory is not sufficient for
all existing commands. A portable artifact-location resolver/export-import path
should preserve original provenance and checksums instead of editing immutable
historical manifests in place. This is a pending implementation, not a supported
feature claimed by this guide.

Before transfer: publish a reviewed code snapshot; add/test portable artifact
resolution; install RoboEnv and the appropriate CPU/CUDA environment on the
other host; transfer one complete small dataset/model bundle; verify checksums
and run a smoke collection/evaluation there; then schedule the larger jobs.
SSH/file-transfer details and the other host's operating system/hardware are not
yet known. No remote connection, upload or deployment has been attempted.
