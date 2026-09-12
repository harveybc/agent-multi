#!/usr/bin/env bash
# R3 (order 2026-09-11): migrate an ALREADY INSTALLED p1lr-decision@
# unit from the ExecStartPre gate check to the ExecCondition gate check.
#
# WHY THIS SCRIPT EXISTS. Correcting the template in the repository is
# not enough. A host that already ran pin_p1lr_decision_runtime.sh or a
# per-campaign drop-in carries `ExecStartPre=<gate check>` in
# ~/.config/systemd/user/p1lr-decision@.service.d/*.conf, and a drop-in
# ExecStartPre= is ADDITIVE: the corrected template would still be
# joined with the old start-pre command, and the restart loop would
# survive the fix.
#
#   bash examples/systemd/repair_p1lr_decision_condition.sh
#
# REVIEWED OPERATOR STEP. It is rootless and idempotent. It:
#   * backs up every drop-in it rewrites, beside the original;
#   * turns each `ExecStartPre=<gate check>` into `ExecCondition=`, and
#     prepends the empty `ExecStartPre=` that clears any inherited one;
#   * reloads the user manager so the change is effective.
#
# It deliberately does NOT: enable, start, restart or stop any unit;
# reconstruct, copy or repoint an absent screen gate; remove an
# installed instance. Removing an obsolete seat is an owner decision
# and this script only prints the exact command for it.
set -euo pipefail

UNIT_DIR="${P1LR_UNIT_DIR:-$HOME/.config/systemd/user}"
DROPIN_DIR="$UNIT_DIR/p1lr-decision@.service.d"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CHANGED=0

if [[ ! -d "$DROPIN_DIR" ]]; then
    echo "no drop-in directory at $DROPIN_DIR — nothing to repair"
else
    shopt -s nullglob
    for conf in "$DROPIN_DIR"/*.conf; do
        grep -q '^ExecStartPre=.\+' "$conf" || continue
        cp -p "$conf" "$conf.pre-r3-$STAMP.bak"
        # 1. every non-empty ExecStartPre becomes an ExecCondition;
        # 2. an empty ExecCondition= is placed before the first of them,
        #    because drop-in Exec* directives ACCUMULATE: without the
        #    reset, the template's condition and every earlier drop-in's
        #    condition would all run, and a stale pinned runtime would be
        #    consulted alongside the current one. The empty line is what
        #    the original `ExecStartPre=` was doing; the conversion has to
        #    carry that meaning across, not just the command;
        # 3. an empty ExecStartPre= is kept (or added) so a start-pre
        #    inherited from the template or an earlier drop-in is cleared
        #    rather than joined.
        sed -i 's|^ExecStartPre=\(.\+\)$|ExecCondition=\1|' "$conf"
        if ! grep -qx 'ExecCondition=' "$conf"; then
            sed -i '0,/^ExecCondition=.\+$/s|^ExecCondition=\(.\+\)$|ExecCondition=\nExecCondition=\1|' \
                "$conf"
        fi
        if ! grep -qx 'ExecStartPre=' "$conf"; then
            sed -i '0,/^\[Service\]$/s|^\[Service\]$|[Service]\nExecStartPre=|' \
                "$conf"
        fi
        CHANGED=$((CHANGED + 1))
        echo "repaired: ${conf/#$HOME/\~}"
    done
    shopt -u nullglob
fi

install -m 0644 "$(dirname "$0")/p1lr-decision@.service" "$UNIT_DIR/"
systemctl --user daemon-reload

echo
echo "drop-ins rewritten: $CHANGED"
echo "effective directives now:"
systemctl --user show 'p1lr-decision@101.service' \
    -p ExecCondition -p ExecStartPre -p Restart \
    -p RestartPreventExitStatus 2>/dev/null \
    | sed "s|$HOME|~|g" | cut -c1-160

cat <<'NOTE'

OWNER DISPOSITION — obsolete P1LR seats.
This script makes an absent or non-viable gate a STABLE refusal: the
unit is skipped, not failed, and systemd schedules no restart. It does
NOT remove a seat that should no longer exist. A seat whose pinned gate
was never produced is historical; to retire it, the owner runs, per
instance:

  systemctl --user disable --now 'p1lr-decision@<seed>.service'
  systemctl --user reset-failed 'p1lr-decision@<seed>.service'

Nothing here starts a new P1LR decision run, and the pinned gate path
is preserved exactly as installed — an absent gate is evidence, not a
file to be recreated.
NOTE
