# Musashi to General Satoshi: Finding-323 Active-Tip Cleanup

DATA-SOTA-340 is accepted. Without GPU work, sanitize or tombstone the 22
registered legacy files on the active public tip, preserve restricted copies by
digest, remove every scan allowlist exemption and produce a zero-exception scan.

Also scan every active public branch tip for real secrets, account identifiers
and personal data. Do not rewrite shared history for topology-only findings. If
an actual secret appears, rotate it first and stop for owner approval of the
smallest viable history rewrite.

Continue WP-PRETRAIN and collectors in parallel; this cleanup does not block CPU
implementation.
