#!/usr/bin/env bash
#
# Install or remove a systemd user service for the goobusters server.
#
# Usage:
#   ./scripts/service.sh            # toggle: install if missing, remove if installed
#   ./scripts/service.sh install     # install & enable the service
#   ./scripts/service.sh remove      # stop, disable & remove the service
#   ./scripts/service.sh status      # show service status
#   ./scripts/service.sh logs        # follow journal logs

set -euo pipefail

SERVICE_NAME="goobusters-server"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_FILE="$UNIT_DIR/$SERVICE_NAME.service"

is_installed() {
    [ -f "$UNIT_FILE" ]
}

cmd_install() {
    mkdir -p "$UNIT_DIR"

    cat > "$UNIT_FILE" <<EOF
[Unit]
Description=Goobusters tracking server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/.venv/bin/python -m lib.server.start -k
Restart=on-failure
RestartSec=5
Environment=VIRTUAL_ENV=$PROJECT_DIR/.venv
Environment=PATH=$PROJECT_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable "$SERVICE_NAME"
    loginctl enable-linger "$USER"
    systemctl --user start "$SERVICE_NAME"
    echo "Installed, enabled, and started $SERVICE_NAME"
    systemctl --user status "$SERVICE_NAME" --no-pager || true
}

cmd_remove() {
    systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true
    systemctl --user disable "$SERVICE_NAME" 2>/dev/null || true
    rm -f "$UNIT_FILE"
    systemctl --user daemon-reload
    echo "Removed $SERVICE_NAME"
}

cmd_status() {
    systemctl --user status "$SERVICE_NAME" --no-pager
}

cmd_logs() {
    journalctl --user -u "$SERVICE_NAME" -f
}

case "${1:-}" in
    install) cmd_install ;;
    remove)  cmd_remove ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    "")
        if is_installed; then
            echo "$SERVICE_NAME is installed, removing..."
            cmd_remove
        else
            echo "$SERVICE_NAME is not installed, installing..."
            cmd_install
        fi
        ;;
    *)
        echo "Usage: $0 {install|remove|status|logs}"
        exit 1
        ;;
esac
