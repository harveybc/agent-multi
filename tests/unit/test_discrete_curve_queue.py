import pytest

from tools.run_discrete_curve_probe import write, receipt_valid
from app.frozen_recipe_inputs import file_sha


def test_resume_authenticates_model_and_identity(tmp_path):
    write(tmp_path / 'weights.json', {'tensor': [1, 2, 3]})
    write(tmp_path / 'RESULT.json', dict(identity='right',
          artifacts={'weights.json': file_sha(tmp_path / 'weights.json')}))
    assert receipt_valid(tmp_path, 'right')
    with pytest.raises(ValueError, match='RESULT_IDENTITY'):
        receipt_valid(tmp_path, 'wrong')
    write(tmp_path / 'weights.json', {'tensor': [1, 2, 4]})
    with pytest.raises(ValueError, match='RESULT_ARTIFACT_CHANGED'):
        receipt_valid(tmp_path, 'right')


def test_no_receipt_is_not_completion(tmp_path):
    write(tmp_path / 'PROGRESS.json', dict(step=32768))
    assert not receipt_valid(tmp_path, 'anything')


def test_no_receipt_path_escape(tmp_path):
    write(tmp_path / 'RESULT.json', dict(identity='right', artifacts={'../weights': 'bad'}))
    with pytest.raises(ValueError, match='RESULT_ARTIFACT_CHANGED'):
        receipt_valid(tmp_path, 'right')
