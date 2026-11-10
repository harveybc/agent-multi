import json
import hashlib
import pytest
from tools.run_stl_cpu_pool import admit, discover_owners, verify_receipt


def host(cap=4,available=10,used=None):
    return dict(cap=cap,reserve=2,reservation=1,available=available,
                memory=used or {},unavailable=False)


def test_completed_and_assigned_are_never_duplicated():
    assigned={'1':'a'}
    new=admit(6,{0},assigned,{'a':host(cap=2),'b':host(cap=2)})
    assert len(new)==3 and not ({0,1}&set(new))
    assert sum(v=='a' for v in new.values())==1
    assert sum(v=='b' for v in new.values())==2
    assert assigned=={'1':'a'}


def test_starting_workers_reserve_their_not_yet_allocated_memory():
    assert admit(4,set(),{'0':'a'},{'a':host(available=3)})=={}
    new=admit(4,set(),{'0':'a'},{'a':host(available=4)})
    assert new=={1:'a'}
    assert admit(4,set(),{'0':'a'},{'a':host(available=3,used={'0':1})})=={1:'a'}


def test_checkpoint_owner_is_sticky_and_ambiguous_owner_fails():
    observations={'a':{'started':[1,2]},'b':{'started':[3]}}
    assert discover_owners(observations,{2},{'1':'a','4':'offline'})=={'1':'a','3':'b','4':'offline'}
    with pytest.raises(ValueError,match='AMBIGUOUS'):
        discover_owners({'a':{'started':[1]},'b':{'started':[1]}},set(),{})
    assert admit(2,set(),{'0':'offline'},{'a':host(cap=1)})=={1:'a'}


def test_unavailable_host_and_insufficient_memory_receive_nothing():
    assert admit(3,set(),{}, {'a':dict(host(),unavailable=True),'b':host(available=2)})=={}


def test_receipts_authenticate_identity_and_bytes(tmp_path):
    (tmp_path/'data').write_bytes(b'original')
    receipt=dict(identity='pinned',files={'data':hashlib.sha256(b'original').hexdigest()})
    (tmp_path/'DONE.json').write_text(json.dumps(receipt))
    verify_receipt(tmp_path,'pinned')
    with pytest.raises(ValueError):verify_receipt(tmp_path,'other')
    (tmp_path/'data').write_bytes(b'corrupted')
    with pytest.raises(ValueError):verify_receipt(tmp_path,'pinned')
