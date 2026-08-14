# Security Policy

## Public repository boundary

This repository is public. Do not commit:

- API keys, passwords, tokens, private keys or signed capabilities;
- broker account identifiers, account fingerprints or live order evidence;
- personal addresses, identity numbers, telephone numbers or dates of birth;
- private-network addresses, Tailscale routes, SSH endpoints or machine-local
  credentials;
- commercial datasets or third-party content that cannot be redistributed.

Runtime authority belongs in ignored local environment files or an operator
configuration directory outside the checkout. Addresses under `192.0.2.0/24`
are documentation examples and do not describe the deployed fleet.

Before publishing, run the repository sensitivity gate:

```bash
python tools/prepush_sensitivity_gate.py --help
cp tools/hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

The hook supplements provider-side secret scanning; it does not make an
exposed credential safe. Revoke and rotate any credential that reaches Git,
then remove it from every published ref and commit.

## Reporting

Do not place suspected secrets or account data in a public issue. Use GitHub's
private vulnerability-reporting channel when available, or contact the
repository owner privately through the GitHub profile.
