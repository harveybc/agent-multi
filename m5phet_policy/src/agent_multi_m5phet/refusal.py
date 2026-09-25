"""The one refusal type this package raises.

It lives in its own module so the observation bridge and the provider can both raise it without
importing each other. `PolicyRefusal` is re-exported from `provider` and from the package root, so
existing imports keep working; this file is plumbing, not a second concept.
"""


class PolicyRefusal(ValueError):
    """A refusal that names itself. Nothing in this package substitutes a default for a refusal."""
