# Social review packet — relocated to private storage (2026-08-10)

The WP5 owner-review packet contains third-party social-media content
(posts, authors, URLs). Publishing scraped third-party content in a
public repository is not authorized; the packet was removed from the
public branch HEAD immediately after detection and now lives in
operator-local private storage:

    ~/.local/state/agent-multi/social-review-private/

Git history still contains the removed blobs (commit 95cb74c0); purging
history is a destructive owner-level action awaiting the owner's
decision. Future review packets are private-by-default.
