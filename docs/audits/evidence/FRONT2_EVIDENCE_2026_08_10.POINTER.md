# Front-2 evidence packets — relocated to private storage (2026-08-10)

The WP4 evidence packets contain operationally sensitive content
(account fingerprints, live paper/demo position and native SL/TP
levels, and the owner's authenticated resume procedure with signer key
paths). Public republication is not authorized; the files were removed
from the public branch HEAD immediately after detection and live in
operator-local private storage:

    ~/.local/state/agent-multi/front2-evidence-private/

Git history still contains the removed blobs (commit c8bcb7a8); purging
history is a destructive owner-level action awaiting the owner's
decision, jointly with the equivalent WP5 removal (42578f70) and the
topology-exposure disposition. Front-2 and security-procedure evidence
is private-by-default from now on.
