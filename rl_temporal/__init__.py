"""Adapters for the SAC/DQN x {native, modular temporal} representation matrix.

Policy training stays in the installed Stable-Baselines3 SAC and DQN through
the existing ``agent_plugins``; this package only plugs a torch
implementation of the modular temporal representation contract
(``predictor.modular.v1`` at engine pin 3ecdb256) into their policies as an
SB3 ``features_extractor``, binds a run to a selected-feature manifest, and
makes save/reload, early stopping, reconciliation and warehouse records
explicit and testable. No RL algorithm is implemented here.
"""

ENGINE_PIN = "3ecdb256"
