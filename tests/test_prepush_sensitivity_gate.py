

class TestHostnameRuleNarrowing:
    """2026-08-16: a QUOTED key with an UNQUOTED value is a variable
    reference in code, never a literal — a literal is always quoted.
    Narrowing that false-positive class must not weaken detection of any
    real literal form."""

    def _finding(self, text: str) -> bool:
        import json as _json
        import pathlib as _pathlib
        from tools.prepush_sensitivity_gate import (
            RULES, _hostname_not_allowed, ALLOWLIST_BASENAME)
        allowlist = _json.loads(
            (_pathlib.Path(__file__).resolve().parent.parent
             / "tools" / ALLOWLIST_BASENAME).read_text())
        rule = next(r for r in RULES if r.name == "hostname_assignment")
        match = rule.regex.search(text)
        return bool(match) and _hostname_not_allowed(match, allowlist)

    def test_quoted_key_with_variable_value_is_not_a_leak(self):
        assert self._finding('"hostname": local_hostname,') is False

    def test_quoted_key_with_unlisted_literal_still_blocks(self):
        assert self._finding('"hostname": "not-an-allowed-host",') is True

    def test_bare_yaml_assignment_still_blocks(self):
        assert self._finding("hostname: not-an-allowed-host") is True

    def test_equals_literal_still_blocks(self):
        assert self._finding('ssh_host = "not-an-allowed-host"') is True

    def test_dynamic_expression_is_not_a_leak(self):
        assert self._finding("hostname: socket.gethostname()") is False
