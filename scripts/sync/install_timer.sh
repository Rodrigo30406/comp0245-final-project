#!/bin/bash
# Nightly push of the light tier (user systemd timer, 03:30 local). Run once on the lab PC after `hf auth login`.
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/comp0245-sync.service <<UNIT
[Unit]
Description=COMP0245 artifact sync (light tier to Hugging Face)
[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
ExecStart=/bin/bash $REPO_ROOT/scripts/sync/push_artifacts.sh
StandardOutput=append:$HOME/.local/state/comp0245-sync.log
StandardError=append:$HOME/.local/state/comp0245-sync.log
UNIT
cat > ~/.config/systemd/user/comp0245-sync.timer <<UNIT
[Unit]
Description=Nightly COMP0245 artifact sync
[Timer]
OnCalendar=*-*-* 03:30:00
Persistent=true
[Install]
WantedBy=timers.target
UNIT
mkdir -p ~/.local/state; systemctl --user daemon-reload; systemctl --user enable --now comp0245-sync.timer
systemctl --user list-timers comp0245-sync.timer --no-pager
