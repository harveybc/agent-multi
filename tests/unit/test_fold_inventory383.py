import json
from docs.work_plan.generalization.fold_inventory383 import audit,end,order_hash

def test_order_dates_and_receipt_presence(tmp_path):
    for name,columns in [('one',['a','b']),('reverse',['b','a'])]:
        root=tmp_path/name
        root.mkdir()
        (root/'RECIPE.json').write_text(json.dumps(dict(list=dict(columns=columns),
            split=dict(periods=dict(TRAIN=['2020-01-01','2022-01-01'])),
            scaler=dict(fit_period=['2020-01-01','2024-01-01']))))
    got=audit([tmp_path],order_hash(['a','b']))
    assert got['counts']['recipe_files']==2
    assert len(got['matches'])==1
    row=got['matches'][0]
    assert (row['actor_train_end'],row['scaler_end'],row['bundle'])==('2022-01-01','2024-01-01',False)
    assert end(['2024-01-01','2024-01-01']) is None

def test_missing_evidence_is_not_an_eligible_model(tmp_path):
    (tmp_path/'RECIPE.json').write_text('{')
    result=audit([tmp_path,tmp_path/'missing'],order_hash(['a']))
    assert not result['matches'] and len(result['inaccessible'])==2
