import pytest
from tools.recipe_sac_queue import write
from app.frozen_recipe_inputs import file_sha
from tools.hybrid_inference_collect import verify_seal


def test_seal_authentication_and_tampering(tmp_path):
    write(tmp_path/'IDENTITY.json',{'test':True})
    write(tmp_path/'report.json',{'profit':-1})
    write(tmp_path/'CLOSE.json',dict(state='COMPLETE',identity_sha256=file_sha(tmp_path/'IDENTITY.json'),
        artifacts={'report.json':file_sha(tmp_path/'report.json')}))
    verify_seal(tmp_path)
    write(tmp_path/'report.json',{'profit':100})
    with pytest.raises(ValueError,match='DIAGNOSTIC_ARTIFACT'):verify_seal(tmp_path)


def test_seal_refuses_escape(tmp_path):
    root=tmp_path/'root';write(root/'IDENTITY.json',{})
    write(tmp_path/'outside.json',{})
    write(root/'CLOSE.json',dict(state='COMPLETE',identity_sha256=file_sha(root/'IDENTITY.json'),
        artifacts={'../outside.json':file_sha(tmp_path/'outside.json')}))
    with pytest.raises(ValueError,match='DIAGNOSTIC_ARTIFACT'):verify_seal(root)
