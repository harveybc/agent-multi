import json
from pathlib import Path

import numpy as np
import pytest
from tools.role464_deap import decode, encode

CFG=json.loads((Path(__file__).resolve().parents[1]/'docs/work_plan/464_deap/config.json').read_text())


@pytest.mark.parametrize('name',CFG['genes'])
def test_endpoints_type_and_baseline(name):
    spec=CFG['genes'][name]
    values=[decode(spec,x) for x in np.linspace(0,1,1001)]
    if spec['kind']=='choice':
        assert set(values)==set(spec['values']) and all(type(v) is int for v in values)
    else:
        assert [values[0],values[-1]]==spec['range'] and np.all(np.diff(values)>0)
    assert decode(spec,encode(spec,spec['baseline']))==pytest.approx(spec['baseline'])


@pytest.mark.parametrize('x',[-.01,1.01,np.nan,np.inf])
def test_invalid_gene_does_not_silently_clip(x):
    with pytest.raises(ValueError):decode(CFG['genes']['atr_window'],x)


def test_no_illegal_sizing():
    assert decode(CFG['genes']['min_rr'],1)<decode(CFG['genes']['full_rr'],0)
