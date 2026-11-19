import pytest
from app.economic_roles385 import best_open

def test_open_symmetry_abstention_and_ties():
    assert best_open(3,1)==1 and best_open(1,3)==2
    assert best_open(-2,0)==0 and best_open(-2,-1)==0
    assert best_open(1,1+1e-10)==-1 and best_open(0,1e-10)==0
    with pytest.raises(ValueError):best_open(float('nan'),2)
