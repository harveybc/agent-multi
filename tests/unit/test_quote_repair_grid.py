import itertools
import json
import math
from pathlib import Path

import pytest

from tools.quote_repair_grid import cases


def design():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/validation/quote_repair_grid.json').read_text())


def test_exact_declared_cartesian_product_and_mirror_pairs():
    cfg=design();cfg['axes']={k:v[:1] for k,v in cfg['axes'].items()}
    cfg['axes']['mirror']=[False,True]
    rows=list(cases(cfg))
    assert len(rows)==2 and [i for i,c in rows]==[0,1]
    assert {k:v for k,v in rows[0][1].items() if k!='mirror'}=={
        k:v for k,v in rows[1][1].items() if k!='mirror'}
    cfg=design()
    count=math.prod(len(v) for v in cfg['axes'].values())
    assert count==1312200
    first=next(cases(cfg))
    assert first[0]==0 and set(first[1])==set(cfg['axes'])


def test_missing_or_duplicate_axis_rejected():
    cfg=design();cfg['axes']['seed']=[7,7]
    with pytest.raises(ValueError):next(cases(cfg))
    cfg=design();del cfg['axes']['mirror']
    with pytest.raises(ValueError):next(cases(cfg))
