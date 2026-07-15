from pathlib import Path

from examples.scripts.install_doin_campaign_supervisor import service_text


def test_service_restarts_supervisor_without_killing_adoptable_workers():
    text = service_text(
        repo=Path("/repo/agent-multi"),
        python=Path("/env/bin/python"),
        profile=Path("/repo/profile.json"),
    )
    assert "Restart=always" in text
    assert "KillMode=process" in text
    assert "-m app.campaign_supervisor --profile /repo/profile.json" in text
